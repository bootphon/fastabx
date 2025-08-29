#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
// #include <torch/csrc/stable/tensor.h> // Fails to build if included here and in dtw.cpp
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <optional>

// Shared memory has a size of 48kB
// Maximum diagonal length is N such that N * 3 * sizeof(float) = 48kB
#define MAX_DIAG_LEN 4096

namespace fastabx {

template <int N>
struct Int64Tuple {
  const int64_t v[N];

  __host__ __device__ const int64_t& operator[](int i) const {
    return v[i];
  }
};

__global__ void dtw_wavefront_kernel(
    float* cost,
    const float* distances,
    const int64_t* sx,
    const int64_t* sy,
    const bool symmetric,
    const Int64Tuple<4> cost_sizes,
    const Int64Tuple<4> cost_strides,
    const Int64Tuple<4> distances_strides) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost_sizes[0] || y >= cost_sizes[1] || (symmetric && x >= y))
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];
  const float* d = distances + (x * distances_strides[0] + y * distances_strides[1]);
  float* c = cost + (x * cost_strides[0] + y * cost_strides[1]);

  __shared__ float buffers[3][MAX_DIAG_LEN];
  int alpha = 0; // Last diagonal
  int beta = 1; // Second to last diagonal
  int gamma = 2; // Buffer for the last diagonal

  for (int64_t diag = 0; diag <= N + M - 1; diag++) {
    const int64_t start_i = min(diag, N - 1);
    const int64_t start_j = max(int64_t(0), diag - start_i);
    const int64_t length = start_i - max(int64_t(0), diag - M + 1) + 1;

    for (int k = threadIdx.x; k < length; k += blockDim.x) {
      const int64_t i = start_i - k;
      const int64_t j = start_j + k;
      const float c_up = (i > 0) ? buffers[alpha][j] : FLT_MAX;
      const float c_left = (j > 0) ? buffers[alpha][j - 1] : FLT_MAX;
      const float c_diag = (i > 0 && j > 0) ? buffers[beta][j - 1] : FLT_MAX;
      const float min_cost = (i == 0 && j == 0) ? 0 : fminf(c_left, fminf(c_diag, c_up));
      const float cij = min_cost + d[i * distances_strides[2] + j * distances_strides[3]];
      c[i * cost_strides[2] + j * cost_strides[3]] = cij;
      buffers[gamma][j] = cij;
    }
    __syncthreads();

    int temp = beta;
    beta = alpha;
    alpha = gamma;
    gamma = temp;
  }
}

__global__ void dtw_backtrack_kernel(
    float* out,
    const float* cost,
    const int64_t* sx,
    const int64_t* sy,
    const bool symmetric,
    const Int64Tuple<2> out_strides,
    const Int64Tuple<4> cost_sizes,
    const Int64Tuple<4> cost_strides) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost_sizes[0] || y >= cost_sizes[1] || (symmetric && x >= y))
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];
  const float* c = cost + (x * cost_strides[0] + y * cost_strides[1]);

  int64_t path_len = 1;
  int64_t i = N - 1;
  int64_t j = M - 1;
  while (i > 0 && j > 0) {
    const float c_up = c[(i - 1) * cost_strides[2] + j * cost_strides[3]];
    const float c_left = c[i * cost_strides[2] + (j - 1) * cost_strides[3]];
    const float c_diag = c[(i - 1) * cost_strides[2] + (j - 1) * cost_strides[3]];
    if (c_diag <= c_left && c_diag <= c_up) {
      i--;
      j--;
    } else if (c_left <= c_up) {
      j--;
    } else {
      i--;
    }
    path_len++;
  }
  if (i == 0)
    path_len += j;
  if (j == 0)
    path_len += i;

  out[x * out_strides[0] + y * out_strides[1]] = c[(N - 1) * cost_strides[2] + (M - 1) * cost_strides[3]] / path_len;
  if (symmetric)
    out[y * out_strides[0] + x * out_strides[1]] = out[x * out_strides[0] + y * out_strides[1]];
}

torch::stable::Tensor dtw_batch_cuda(
    const torch::stable::Tensor distances,
    const torch::stable::Tensor sx,
    const torch::stable::Tensor sy,
    bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const int64_t max_x = distances.size(2);
  const int64_t max_y = distances.size(3);

  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  STD_TORCH_CHECK(max_x < MAX_DIAG_LEN, "Diagonal too large to use CUDA shared memory");

  torch::stable::Tensor cost = torch::stable::new_zeros(distances, {nx, ny, max_x, max_y});
  torch::stable::Tensor out = torch::stable::new_empty(distances, {nx, ny});

  const dim3 num_blocks(nx, ny);
  const int num_threads = max_x > 1024 ? 1024 : max_x;
  torch::stable::accelerator::DeviceIndex device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  cudaStream_t stream = (cudaStream_t)torch::stable::accelerator::getCurrentStream(device_idx).id();

  dtw_wavefront_kernel<<<num_blocks, num_threads, 0, stream>>>(
      reinterpret_cast<float*>(cost.data_ptr()),
      reinterpret_cast<const float*>(distances.data_ptr()),
      reinterpret_cast<int64_t*>(sx.data_ptr()),
      reinterpret_cast<int64_t*>(sy.data_ptr()),
      symmetric,
      {nx, ny, max_x, max_y},
      {cost.stride(0), cost.stride(1), cost.stride(2), cost.stride(3)},
      {distances.stride(0), distances.stride(1), distances.stride(2), distances.stride(3)});
  dtw_backtrack_kernel<<<num_blocks, 1, 0, stream>>>(
      reinterpret_cast<float*>(out.data_ptr()),
      reinterpret_cast<float*>(cost.data_ptr()),
      reinterpret_cast<int64_t*>(sx.data_ptr()),
      reinterpret_cast<int64_t*>(sy.data_ptr()),
      symmetric,
      {out.stride(0), out.stride(1)},
      {nx, ny, max_x, max_y},
      {cost.stride(0), cost.stride(1), cost.stride(2), cost.stride(3)});
  return out;
}

torch::stable::Tensor dtw_cuda(const torch::stable::Tensor distances) {
  torch::stable::Tensor sx =
      torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  torch::stable::fill_(sx, distances.size(0));
  torch::stable::Tensor sy =
      torch::stable::new_empty(distances, {1}, std::make_optional(torch::headeronly::ScalarType::Long));
  torch::stable::fill_(sy, distances.size(1));

  AtenTensorHandle distances_ath;
  const int64_t shape[4] = {1, 1, distances.size(0), distances.size(1)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_reshape(distances.get(), shape, 4, &distances_ath));
  const torch::stable::Tensor distances_resized = torch::stable::Tensor(distances_ath);
  torch::stable::Tensor result = dtw_batch_cuda(distances_resized, sx, sy, false);

  AtenTensorHandle out_ath;
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_squeeze_dim(result.get(), 0, &out_ath));
  TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_squeeze_dim(out_ath, 0, &out_ath));
  return torch::stable::Tensor(out_ath);
}

void boxed_dtw_cuda(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  stack[0] = from(dtw_cuda(to<torch::stable::Tensor>(stack[0])));
}

void boxed_dtw_batch_cuda(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  stack[0] = from(dtw_batch_cuda(
      to<torch::stable::Tensor>(stack[0]),
      to<torch::stable::Tensor>(stack[1]),
      to<torch::stable::Tensor>(stack[2]),
      to<bool>(stack[3])));
}

STABLE_TORCH_LIBRARY_IMPL(fastabx, CUDA, m) {
  m.impl("dtw", &boxed_dtw_cuda);
  m.impl("dtw_batch", &boxed_dtw_batch_cuda);
}

} // namespace fastabx

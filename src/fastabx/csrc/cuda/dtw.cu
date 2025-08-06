#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

// Shared memory has a size of 48kB
// Maximum diagonal length is N such that N * 3 * sizeof(float) = 48kB
#define MAX_DIAG_LEN 4096

using torch::stable::Tensor;

namespace fastabx {

__global__ void dtw_wavefront_kernel(
    float* cost,
    const float* distances,
    const int64_t* sx,
    const int64_t* sy,
    bool symmetric,
    int64_t cost_sizes_0,
    int64_t cost_sizes_1,
    int64_t cost_strides_0,
    int64_t cost_strides_1,
    int64_t cost_strides_2,
    int64_t cost_strides_3,
    int64_t distances_strides_0,
    int64_t distances_strides_1,
    int64_t distances_strides_2,
    int64_t distances_strides_3) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost_sizes_0 || y >= cost_sizes_1)
    return;
  if (symmetric && x >= y)
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];

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
      const float min_cost = (i == 0 && j == 0) ? 0 : min(c_left, min(c_diag, c_up));
      const float cij = min_cost +
          distances[x * distances_strides_0 + y * distances_strides_1 + i * distances_strides_2 +
                    j * distances_strides_3];
      cost[x * cost_strides_0 + y * cost_strides_1 + i * cost_strides_2 + j * cost_strides_3] = cij;
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
    bool symmetric,
    int64_t out_strides_0,
    int64_t out_strides_1,
    int64_t cost_sizes_0,
    int64_t cost_sizes_1,
    int64_t cost_strides_0,
    int64_t cost_strides_1,
    int64_t cost_strides_2,
    int64_t cost_strides_3) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost_sizes_0 || y >= cost_sizes_1)
    return;
  if (symmetric && x >= y)
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];

  int64_t path_len = 1;
  int64_t i = N - 1;
  int64_t j = M - 1;
  while (i > 0 && j > 0) {
    const float c_up = cost[x * cost_strides_0 + y * cost_strides_1 + (i - 1) * cost_strides_2 + j * cost_strides_3];
    const float c_left = cost[x * cost_strides_0 + y * cost_strides_1 + i * cost_strides_2 + (j - 1) * cost_strides_3];
    const float c_diag =
        cost[x * cost_strides_0 + y * cost_strides_1 + (i - 1) * cost_strides_2 + (j - 1) * cost_strides_3];
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

  out[x * out_strides_0 + y * out_strides_1] =
      cost[x * cost_strides_0 + y * cost_strides_1 + (N - 1) * cost_strides_2 + (M - 1) * cost_strides_3] / path_len;
  if (symmetric)
    out[y * out_strides_0 + x * out_strides_1] = out[x * out_strides_0 + y * out_strides_1];
}

Tensor dtw_batch_cuda(const Tensor distances, const Tensor sx, const Tensor sy, bool symmetric) {
  const auto nx = distances.size(0);
  const auto ny = distances.size(1);
  const auto max_x = distances.size(2);
  const auto max_y = distances.size(3);

  STD_TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  STD_TORCH_CHECK(max_x < MAX_DIAG_LEN, "Diagonal too large to use CUDA shared memory");

  auto cost = empty_like(distances);
  zero_(cost);
  int64_t sizes[2] = {nx, ny};
  int64_t strides[2] = {ny, 1};
  AtenTensorHandle ath;
  aoti_torch_empty_strided(
      2, sizes, strides, aoti_torch_dtype_float32(), aoti_torch_device_type_cuda(), distances.get_device(), &ath);
  auto out = Tensor(ath);

  const dim3 num_blocks(nx, ny);
  const int num_threads = max_x > 1024 ? 1024 : max_x;

  dtw_wavefront_kernel<<<num_blocks, num_threads>>>(
      reinterpret_cast<float*>(cost.data_ptr()),
      reinterpret_cast<const float*>(distances.data_ptr()),
      reinterpret_cast<const int64_t*>(sx.data_ptr()),
      reinterpret_cast<const int64_t*>(sy.data_ptr()),
      symmetric,
      nx,
      ny,
      cost.stride(0),
      cost.stride(1),
      cost.stride(2),
      cost.stride(3),
      distances.stride(0),
      distances.stride(1),
      distances.stride(2),
      distances.stride(3));
  dtw_backtrack_kernel<<<num_blocks, 1>>>(
      reinterpret_cast<float*>(out.data_ptr()),
      reinterpret_cast<const float*>(cost.data_ptr()),
      reinterpret_cast<const int64_t*>(sx.data_ptr()),
      reinterpret_cast<const int64_t*>(sy.data_ptr()),
      symmetric,
      out.stride(0),
      out.stride(1),
      nx,
      ny,
      cost.stride(0),
      cost.stride(1),
      cost.stride(2),
      cost.stride(3));
  return out;
}

Tensor dtw_cuda(const Tensor distances) {
  int64_t tensor_size[1] = {1};
  int64_t tensor_stride[1] = {1};

  AtenTensorHandle sx_ath;
  aoti_torch_empty_strided(
      1,
      tensor_size,
      tensor_stride,
      aoti_torch_dtype_int64(),
      aoti_torch_device_type_cuda(),
      distances.get_device(),
      &sx_ath);
  auto sx = Tensor(sx_ath);
  aoti_torch_cuda_fill__Scalar(sx.get(), distances.size(0));

  AtenTensorHandle sy_ath;
  aoti_torch_empty_strided(
      1,
      tensor_size,
      tensor_stride,
      aoti_torch_dtype_int64(),
      aoti_torch_device_type_cuda(),
      distances.get_device(),
      &sy_ath);
  auto sy = Tensor(sy_ath);
  aoti_torch_cuda_fill__Scalar(sy.get(), distances.size(1));

  AtenTensorHandle distances_ath;
  int64_t shape[4] = {1, 1, distances.size(0), distances.size(1)};
  aoti_torch_cuda_reshape(distances.get(), shape, 4, &distances_ath);
  auto distances_resized = Tensor(distances_ath);
  auto result = dtw_batch_cuda(distances_resized, sx, sy, false);

  AtenTensorHandle out_ath;
  aoti_torch_cuda_squeeze_dim(result.get(), 0, &out_ath);
  aoti_torch_cuda_squeeze_dim(out_ath, 0, &out_ath);
  return Tensor(out_ath);
}

void boxed_dtw_cuda(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  stack[0] = from(dtw_cuda(to<Tensor>(stack[0])));
}

void boxed_dtw_batch_cuda(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  auto result = dtw_batch_cuda(to<Tensor>(stack[0]), to<Tensor>(stack[1]), to<Tensor>(stack[2]), to<bool>(stack[3]));
  stack[0] = from(result);
}

STABLE_TORCH_LIBRARY_IMPL(fastabx, CUDA, m) {
  m.impl("dtw", &boxed_dtw_cuda);
  m.impl("dtw_batch", &boxed_dtw_batch_cuda);
}

} // namespace fastabx

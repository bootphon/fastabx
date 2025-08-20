#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

// Shared memory has a size of 48kB
// Maximum diagonal length is N such that N * 3 * sizeof(float) = 48kB
#define MAX_DIAG_LEN 4096

namespace fastabx {

__global__ void dtw_wavefront_kernel(
    torch::PackedTensorAccessor32<float, 4> cost,
    const torch::PackedTensorAccessor32<float, 4> distances,
    const torch::PackedTensorAccessor32<int64_t, 1> sx,
    const torch::PackedTensorAccessor32<int64_t, 1> sy,
    bool symmetric) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost.size(0) || y >= cost.size(1))
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
      const float cij = distances[x][y][i][j] + min_cost;
      cost[x][y][i][j] = cij;
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
    torch::PackedTensorAccessor32<float, 2> out,
    const torch::PackedTensorAccessor32<float, 4> cost,
    const torch::PackedTensorAccessor32<int64_t, 1> sx,
    const torch::PackedTensorAccessor32<int64_t, 1> sy,
    bool symmetric) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  if (x >= cost.size(0) || y >= cost.size(1))
    return;
  if (symmetric && x >= y)
    return;
  const int64_t N = sx[x];
  const int64_t M = sy[y];

  int64_t path_len = 1;
  int64_t i = N - 1;
  int64_t j = M - 1;
  while (i > 0 && j > 0) {
    const float c_up = cost[x][y][i - 1][j];
    const float c_left = cost[x][y][i][j - 1];
    const float c_diag = cost[x][y][i - 1][j - 1];
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

  out[x][y] = cost[x][y][N - 1][M - 1] / path_len;
  if (symmetric)
    out[y][x] = out[x][y];
}

torch::Tensor dtw_cuda(torch::Tensor distances) {
  // Note: the warning raised here won't be converted to a Python UserWarning.
  // This could be done with the HANDLE_TH_ERRORS macro, but since it uses pybind11,
  // the extension cannot be compiled with the Limited API anymore.
  TORCH_WARN_ONCE("The 2D DTW implementation is usually faster on CPU than on GPU")
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
  const auto options_long = torch::TensorOptions().dtype(torch::kInt64).device(distances.device());
  const auto N = distances.size(0);
  const auto M = distances.size(1);
  const auto sx = torch::tensor({N}, options_long);
  const auto sy = torch::tensor({M}, options_long);
  const auto distances_unsqueezed = distances.view({1, 1, N, M});
  auto cost = torch::zeros({1, 1, N, M}, options);
  auto out = torch::zeros({1, 1}, options);

  TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  TORCH_CHECK(N < MAX_DIAG_LEN, "Diagonal too large to use CUDA shared memory");
  const int num_threads = N > 1024 ? 1024 : N;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dtw_wavefront_kernel<<<1, num_threads, 0, stream>>>(
      cost.packed_accessor32<float, 4>(),
      distances_unsqueezed.packed_accessor32<float, 4>(),
      sx.packed_accessor32<int64_t, 1>(),
      sy.packed_accessor32<int64_t, 1>(),
      false);
  dtw_backtrack_kernel<<<1, 1, 0, stream>>>(
      out.packed_accessor32<float, 2>(),
      cost.packed_accessor32<float, 4>(),
      sx.packed_accessor32<int64_t, 1>(),
      sy.packed_accessor32<int64_t, 1>(),
      false);
  return out.squeeze();
}

torch::Tensor dtw_batch_cuda(torch::Tensor distances, torch::Tensor sx, torch::Tensor sy, bool symmetric) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
  const auto nx = distances.size(0);
  const auto ny = distances.size(1);
  const auto max_x = distances.size(2);
  const auto max_y = distances.size(3);
  auto cost = torch::zeros({nx, ny, max_x, max_y}, options);
  auto out = torch::zeros({nx, ny}, options);

  TORCH_CHECK(nx > 0 && ny > 0 && max_x > 0 && max_y > 0, "Empty input tensor");
  TORCH_CHECK(max_x < MAX_DIAG_LEN, "Diagonal too large to use CUDA shared memory");
  const dim3 num_blocks(nx, ny);
  const int num_threads = max_x > 1024 ? 1024 : max_x;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dtw_wavefront_kernel<<<num_blocks, num_threads, 0, stream>>>(
      cost.packed_accessor32<float, 4>(),
      distances.packed_accessor32<float, 4>(),
      sx.packed_accessor32<int64_t, 1>(),
      sy.packed_accessor32<int64_t, 1>(),
      symmetric);
  dtw_backtrack_kernel<<<num_blocks, 1, 0, stream>>>(
      out.packed_accessor32<float, 2>(),
      cost.packed_accessor32<float, 4>(),
      sx.packed_accessor32<int64_t, 1>(),
      sy.packed_accessor32<int64_t, 1>(),
      symmetric);
  return out;
}

TORCH_LIBRARY_IMPL(fastabx, CUDA, m) {
  m.impl("dtw", &dtw_cuda);
  m.impl("dtw_batch", &dtw_batch_cuda);
}

} // namespace fastabx

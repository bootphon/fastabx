#include <Python.h>
#include <omp.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <algorithm>
#include <vector>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the STABLE_TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1, /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

using torch::stable::Tensor;

namespace fastabx {

inline float dtw(
    const float* distances,
    const int64_t N,
    const int64_t M,
    const int64_t stride_x,
    const int64_t stride_y) {
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  STD_TORCH_CHECK(stride_x > 0 && stride_y > 0, "Strides must be positive");
  std::vector<float> cost(N * M);

  cost[0] = distances[0];
  for (int64_t i = 1; i < N; i++) {
    cost[i * M] = distances[i * stride_x] + cost[(i - 1) * M];
  }
  for (int64_t j = 1; j < M; j++) {
    cost[j] = distances[j * stride_y] + cost[j - 1];
  }
  for (int64_t i = 1; i < N; i++) {
    for (int64_t j = 1; j < M; j++) {
      cost[i * M + j] = distances[i * stride_x + j * stride_y] +
          std::min({cost[(i - 1) * M + j], cost[(i - 1) * M + j - 1], cost[i * M + j - 1]});
    }
  }

  int64_t path_len = 1;
  int64_t i = N - 1;
  int64_t j = M - 1;
  while (i > 0 && j > 0) {
    const float c_up = cost[(i - 1) * M + j];
    const float c_left = cost[i * M + j - 1];
    const float c_diag = cost[(i - 1) * M + j - 1];
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
  return cost[(N - 1) * M + M - 1] / path_len;
}

Tensor dtw_cpu(const Tensor distances) {
  float result =
      dtw(reinterpret_cast<const float*>(distances.data_ptr()),
          distances.size(0),
          distances.size(1),
          distances.stride(0),
          distances.stride(1));
  AtenTensorHandle ath;
  aoti_torch_scalar_to_tensor_float32(result, &ath);
  return Tensor(ath);
}

Tensor dtw_batch_cpu(const Tensor distances, const Tensor sx, const Tensor sy, bool symmetric) {
  const int64_t nx = distances.size(0);
  const int64_t ny = distances.size(1);
  const float* distances_ptr = reinterpret_cast<const float*>(distances.data_ptr());
  const int64_t* sx_ptr = reinterpret_cast<const int64_t*>(sx.data_ptr());
  const int64_t* sy_ptr = reinterpret_cast<const int64_t*>(sy.data_ptr());

  const int64_t sizes[2] = {nx, ny};
  const int64_t strides[2] = {ny, 1};
  AtenTensorHandle ath;
  aoti_torch_empty_strided(2, sizes, strides, aoti_torch_dtype_float32(), aoti_torch_device_type_cpu(), 0, &ath);
  Tensor out = Tensor(ath);
  zero_(out);
  float* out_ptr = reinterpret_cast<float*>(out.data_ptr());

#pragma omp parallel for schedule(dynamic)
  for (int64_t i = 0; i < nx; i++) {
    const int64_t start_j = symmetric ? i : 0;
    for (int64_t j = start_j; j < ny; j++) {
      if (symmetric && i == j)
        continue;
      out_ptr[i * ny + j] =
          dtw(distances_ptr + i * distances.stride(0) + j * distances.stride(1),
              sx_ptr[i],
              sy_ptr[j],
              distances.stride(2),
              distances.stride(3));
      if (symmetric && i != j) {
        out_ptr[j * ny + i] = out_ptr[i * ny + j];
      }
    }
  };
  return out;
}

void boxed_dtw_cpu(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  stack[0] = from(dtw_cpu(to<Tensor>(stack[0])));
}

void boxed_dtw_batch_cpu(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  stack[0] = from(dtw_batch_cpu(to<Tensor>(stack[0]), to<Tensor>(stack[1]), to<Tensor>(stack[2]), to<bool>(stack[3])));
}

STABLE_TORCH_LIBRARY(fastabx, m) {
  m.def("dtw(Tensor distances) -> Tensor");
  m.def("dtw_batch(Tensor distances, Tensor sx, Tensor sy, bool symmetric) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(fastabx, CPU, m) {
  m.impl("dtw", &boxed_dtw_cpu);
  m.impl("dtw_batch", &boxed_dtw_batch_cpu);
}

} // namespace fastabx

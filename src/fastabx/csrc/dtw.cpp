#include <Python.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <algorithm>
#include <iostream>
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

inline float dtw(const float* distances, int64_t N, int64_t M) {
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  std::vector<float> cost(N * M);

  cost[0] = distances[0];
  for (int64_t i = 1; i < N; i++) {
    cost[i * M] = distances[i * M] + cost[(i - 1) * M];
  }
  for (int64_t j = 1; j < M; j++) {
    cost[j] = distances[j] + cost[j - 1];
  }
  for (int64_t i = 1; i < N; i++) {
    for (int64_t j = 1; j < M; j++) {
      cost[i * M + j] =
          distances[i * M + j] + std::min({cost[(i - 1) * M + j], cost[(i - 1) * M + j - 1], cost[i * M + j - 1]});
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
  float result = dtw(reinterpret_cast<const float*>(distances.data_ptr()), distances.size(0), distances.size(1));
  AtenTensorHandle ath;
  aoti_torch_scalar_to_tensor_float32(result, &ath);
  return Tensor(ath);
}

Tensor dtw_batch_cpu(const Tensor distances, const Tensor sx, const Tensor sy, bool symmetric) {
  const auto nx = distances.size(0);
  const auto ny = distances.size(1);
  const auto sx_a = reinterpret_cast<const int64_t*>(sx.data_ptr());
  const auto sy_a = reinterpret_cast<const int64_t*>(sy.data_ptr());

  int64_t sizes[2] = {nx, ny};
  int64_t strides[2] = {distances.stride(0), distances.stride(1)};
  int32_t dtype;
  aoti_torch_get_dtype(distances.get(), &dtype);
  int32_t device_type;
  aoti_torch_get_device_type(distances.get(), &device_type);
  AtenTensorHandle ath;
  aoti_torch_empty_strided(2, sizes, strides, dtype, device_type, distances.get_device(), &ath);
  auto out = Tensor(ath);
  auto out_a = reinterpret_cast<float*>(out.data_ptr());

  for (int64_t i = 0; i < nx; i++) {
    const int64_t start_j = symmetric ? i : 0;
    for (int64_t j = start_j; j < ny; j++) {
      if (symmetric && i == j)
        continue;

      AtenTensorHandle sub_ath;
      AtenTensorHandle sub_ath2;
      int64_t i_end = i + 1;
      int64_t j_end = j + 1;
      aoti_torch_cpu_slice_Tensor(distances.get(), 0, &i, &i_end, 1, &sub_ath);
      aoti_torch_cpu_slice_Tensor(sub_ath, 1, &j, &j_end, 1, &sub_ath2);
      const auto sub_distances = Tensor(sub_ath2);

      out_a[i * ny + j] = dtw(reinterpret_cast<const float*>(sub_distances.data_ptr()), sx_a[i], sy_a[j]);
      if (symmetric && i != j) {
        out_a[j * ny + i] = out_a[i * ny + j];
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

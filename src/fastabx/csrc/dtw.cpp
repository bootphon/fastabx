#include <omp.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/types.h>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
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

namespace fastabx {

float _dtw_cpu(torch::Tensor distances) {
  const auto N = distances.size(0);
  const auto M = distances.size(1);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
  const auto distances_a = distances.accessor<float, 2>();
  auto cost = torch::zeros({N, M}, options);
  auto cost_a = cost.accessor<float, 2>();

  cost_a[0][0] = distances_a[0][0];
  for (int64_t i = 1; i < N; i++) {
    cost_a[i][0] = distances_a[i][0] + cost_a[i - 1][0];
  }
  for (int64_t j = 1; j < M; j++) {
    cost_a[0][j] = distances_a[0][j] + cost_a[0][j - 1];
  }
  for (int64_t i = 1; i < N; i++) {
    for (int64_t j = 1; j < M; j++) {
      cost_a[i][j] = distances_a[i][j] + std::min({cost_a[i - 1][j], cost_a[i - 1][j - 1], cost_a[i][j - 1]});
    }
  }

  int64_t path_len = 1;
  int64_t i = N - 1;
  int64_t j = M - 1;
  while (i > 0 && j > 0) {
    const float c_up = cost_a[i - 1][j];
    const float c_left = cost_a[i][j - 1];
    const float c_diag = cost_a[i - 1][j - 1];
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
  return cost_a[N - 1][M - 1] / path_len;
}

torch::Tensor dtw_cpu(torch::Tensor distances) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
  return torch::tensor(_dtw_cpu(distances), options);
}

torch::Tensor dtw_batch_cpu(torch::Tensor distances, torch::Tensor sx, torch::Tensor sy, bool symmetric) {
  const auto nx = distances.size(0);
  const auto ny = distances.size(1);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
  const auto sx_a = sx.accessor<int64_t, 1>();
  const auto sy_a = sy.accessor<int64_t, 1>();
  auto out = torch::zeros({nx, ny}, options);
  auto out_a = out.accessor<float, 2>();

#pragma omp parallel for schedule(dynamic)
  for (int64_t i = 0; i < nx; i++) {
    const int64_t start_j = symmetric ? i : 0;
    for (int64_t j = start_j; j < ny; j++) {
      if (symmetric && i == j)
        continue;
      const auto sub_distances = distances.index({i, j, torch::indexing::Slice(), torch::indexing::Slice()})
                                     .slice(0, 0, sx_a[i])
                                     .slice(1, 0, sy_a[j]);
      out_a[i][j] = _dtw_cpu(sub_distances);
      if (symmetric && i != j) {
        out_a[j][i] = out_a[i][j];
      }
    }
  }
  return out;
}

TORCH_LIBRARY(fastabx, m) {
  m.def("dtw(Tensor distances) -> Tensor", {torch::Tag::pt2_compliant_tag});
  m.def("dtw_batch(Tensor distances, Tensor sx, Tensor sy, bool symmetric) -> Tensor", {torch::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(fastabx, CPU, m) {
  m.impl("dtw", &dtw_cpu);
  m.impl("dtw_batch", &dtw_batch_cpu);
}

} // namespace fastabx

# Fast ABX evaluation

[![PyPI](https://img.shields.io/pypi/v/fastabx)](https://pypi.org/project/fastabx/)
[![Python](https://img.shields.io/pypi/pyversions/fastabx)](https://pypi.org/project/fastabx/)
[![CI](https://github.com/bootphon/fastabx/actions/workflows/ci.yml/badge.svg)](https://github.com/bootphon/fastabx/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-fastabx-blue)](https://docs.cognitive-ml.fr/fastabx)
[![License](https://img.shields.io/pypi/l/fastabx)](https://github.com/bootphon/fastabx/blob/main/LICENSE)

**fastabx** is a Python package for efficient computation of ABX discriminability.

The ABX discriminability measures how well categories of interest are separated in the representation space by
determining whether tokens from the same category are closer to each other than to those from a different category.
While ABX has been mostly used to evaluate speech representations, it is a generic framework that can be applied
to other domains of representation learning.

This package provides a simple interface that can be adapted to any ABX conditions, and to any input modality.

- **Generic**: any ON, BY and ACROSS conditions, on dense or discrete representations, in any modality.
- **Fast**: triplets are built with lazy [polars](https://github.com/pola-rs/polars) queries, and distances are
  computed in batch on CPU or GPU, with DTW running as a PyTorch C++/CUDA extension.
- **Ready to use**: [`zerospeech_abx`](https://docs.cognitive-ml.fr/fastabx/api.html) and the `fastabx` command
  line interface reproduce the ZeroSpeech triphone and phoneme ABX out of the box.

Check out the documentation for more information: https://docs.cognitive-ml.fr/fastabx

## Install

Install the pre-built package in your environment:

```bash
pip install fastabx
```

It requires Python 3.12 or later, and depends on PyTorch 2.10.0 or later, NumPy, Polars, tqdm, and [torchdtw](https://github.com/bootphon/torchdtw).

## Quickstart

Simple example:

```python
import numpy as np
from fastabx import Dataset, Score, Task

rng = np.random.default_rng(0)
features = np.concatenate([rng.normal(0, 1, (50, 8)), rng.normal(2, 1, (50, 8))])
labels = {"phone": ["a"] * 50 + ["b"] * 50, "speaker": ["s1", "s2"] * 50}

dataset = Dataset.from_numpy(features, labels)  # What to compare
task = Task(dataset, on="phone", by=["speaker"])  # Which triplets to build
score = Score(task, "euclidean")  # How to compare them

print(score.collapse(levels=["speaker"]))  # ABX error rate
# 0.03059999644756317
```

On speech, build the dataset from an item file and a directory of features instead, then run the same
`Task` / `Score` pipeline:

```python
from fastabx import Dataset

dataset = Dataset.from_item("./triphone-dev-clean.item", "./hubert-l11-dev-clean", frequency=50)
task = Task(dataset, on="#phone", by=["speaker", "next-phone", "prev-phone"])
score = Score(task, "angular")
print(score.collapse(levels=[("next-phone", "prev-phone"), "speaker"]))
```

The standard ZeroSpeech evaluation is available as a single function, and as a CLI:

```python
from fastabx import zerospeech_abx

error_rate = zerospeech_abx(
    "./triphone-dev-clean.item",
    "./hubert-l11-dev-clean",
    max_size_group=10,
    speaker="within",
    context="within",
)
```

```bash
fastabx ./triphone-dev-clean.item ./hubert-l11-dev-clean --max-size-group 10
# ABX error rate: 3.378%
```

Scores are ABX error rates: lower is better and chance is 0.5.

See the [user guide](https://docs.cognitive-ml.fr/fastabx/guide.html) for the full pipeline, the
[API reference](https://docs.cognitive-ml.fr/fastabx/api.html) for every option, and the
[examples](https://docs.cognitive-ml.fr/fastabx/examples/index.html) for subsampling, pooling and constraints.

## Citation

A preprint is available on arXiv: https://arxiv.org/abs/2505.02692 \
If you use fastabx in your work, please cite it:

```bibtex
@misc{fastabx,
  title={fastabx: A library for efficient computation of ABX discriminability},
  author={Maxime Poli and Emmanuel Chemla and Emmanuel Dupoux},
  year={2025},
  eprint={2505.02692},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.02692},
}
```

## License

fastabx is released under the [MIT license](LICENSE).

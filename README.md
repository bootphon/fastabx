# Fast ABX evaluation

**fastabx** is a Python package for efficient computation of ABX discriminability.

The ABX discriminability measures how well categories of interest are separated in the representation space by
determining whether tokens from the same category are closer to each other than to those from a different category.
While ABX has been mostly used to evaluate speech representations, it is a generic framework that can be applied
to other domains of representation learning.

This package provides a simple interface that can be adapted to any ABX conditions, and to any input modality.

Check out the documentation for more information: https://docs.cognitive-ml.fr/fastabx

## Install

Install the pre-built package in your environment:

```bash
pip install fastabx
```

It requires Python 3.12 or later and the default PyTorch version on PyPI (2.8.0, CUDA 12.8 variant for Linux, CPU variant for Windows and macOS).
Wheels compatible with other versions and variants of PyTorch are available on the GitHub Releases page.

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

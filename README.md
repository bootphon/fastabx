# Fast ABX evaluation

**fastabx** is a Python package for efficient computation of ABX discriminability.

The ABX discriminability is measures how well categories of interest are separated in the representation space by
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

It requires Python 3.12 or later, and PyTorch 2.6.0 (CUDA 12.4 variant for Linux and Windows).
Wheels are available for Linux x86-64 (glibc 2.34 or later), macOS 14 or later, and Windows x86-64.

## Citation

If you use fastabx in your work, please cite the following preprint (available here: https://arxiv.org/abs/2505.02692):

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

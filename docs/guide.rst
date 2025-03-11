==========
User guide
==========

ZeroSpeech 2021 ABX task
========================

Python API
----------

.. code-block:: python

    import torch
    from fastabx import Dataset, Score, Subsampler, Task


    item, features = "./triphone-dev-clean.item", "./features"
    frequency, maker = 50, torch.load

    dataset = Dataset.from_item(item, features, frequency, maker)
    task = Task(dataset, on="#phone", by=["next-phone", "prev-phone", "speaker"], subsampler=Subsampler())
    score = Score(task, "cosine")
    print(score.collapse(levels=[("next-phone", "prev-phone"), "speaker"]))

CLI
---

.. code-block:: console

    ❯ fastabx --help
    usage: fastabx [-h] [--frequency FREQUENCY] [--speaker {within,across}] [--context {within,any}]
                   [--distance {euclidean,cosine,angular,kl,kl_symmetric,identical,null}] [--max-size-group MAX_SIZE_GROUP]
                   [--max-x-across MAX_X_ACROSS] [--seed SEED]
                   item features

    ZeroSpeech ABX

    positional arguments:
      item                  Path to the item file
      features              Path to the features directory

    options:
      -h, --help            show this help message and exit
      --frequency FREQUENCY
                            Feature frequency (in Hz) (default: 50)
      --speaker {within,across}
                            Speaker mode (default: within)
      --context {within,any}
                            Context mode (default: within)
      --distance {euclidean,cosine,angular,kl,kl_symmetric,identical,null}
                            Distance (default: cosine)
      --max-size-group MAX_SIZE_GROUP
                            Maximum number of A, B, or X in a cell (default: 10)
      --max-x-across MAX_X_ACROSS
                            With 'across', maximum number of X given (A, B) (default: 5)
      --seed SEED           Random seed (default: 0)

Motivation
==========

1. Simple and generic API
2. As fast as possible

This library aims to be as clear and minimal as possible to make its maintenance easy,
and the code readable and quick to understand. It should be easy to incorporate
different components into one's personal code, and not just use it as a black box.

At the same time, it must be as fast as possible to calculate the ABX, both in
forming triplets and calculating the distances themselves, while offering the
possibility to use any configuration of "on," "by," and "across" conditions.

The idea of creating yet again a new ABX library comes from the realization
that https://github.com/pola-rs/polars efficiently and easily
solves the difficulties associated with creating triplets.

We can write the creation of the triplets as some "join" and "select" operations
on dataframes, then some "filter" for subsampling. With `polars`, the full query
is built lazily and then processed end-to-end. The backend will run several
optimizations for us, and can even run on GPU. We don't have to worry anymore
about how to built the triplets in a clever manner.

The computation of the distances is similar as
https://github.com/zerospeech/libri-light-abx2.
The distances functions have been modified to be more memory
efficient by avoiding large broadcastings.
The important change is that now the DTW is computed in a PyTorch C++ extension,
with CPU (using OpenMP) and CUDA backends. The speedup is most noticeable on
large cells, such as those obtained when running the Phoneme ABX without
context conditions.

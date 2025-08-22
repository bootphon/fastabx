==========
User guide
==========

| If you are coming from ABXpy or Libri-light / ZeroSpeech 2021 ABX, see :ref:`this page <other libs>`.
| If you want to reproduce experiments from the paper, see: https://github.com/mxmpl/fastabx-paper.

Python API
==========

The library provides one function that can be used out of the box: :func:`.zerospeech_abx`.
This function computes the triphone or phoneme ABX, similarly as in past ZeroSpeech challenges.
It is also available through a command line interface.

.. code-block:: python

   from fastabx import zerospeech_abx

   item, features, frequency = "./triphone-dev-clean.item", "./hubert-l11-dev-clean", 50
   abx_error_rate = zerospeech_abx(
       item,
       features,
       max_size_group=10,
       max_x_across=5,
       speaker="within",
       context="within",
       distance="angular",
       frequency=frequency,
       seed=0,
   )
   print(abx_error_rate)
   # 0.033783210627340875


The main interface of the library consists of three classes: :class:`.Dataset`, :class:`.Task`, and :class:`.Score`.
The :class:`.Dataset` is a simple wrapper to the underlying corpus: it is made of labels and of a way to access the
representations. We provide several class methods to create a Dataset from arrays, CSV files, or using
an item file and a function to extract representations.

.. code-block:: python

   from fastabx import Dataset

   item, features, frequency = "./triphone-dev-clean.item", "./hubert-l11-dev-clean", 50
   dataset = Dataset.from_item(item, features, frequency)

The ABX :class:`.Task` is build given a :class:`.Dataset` and the ON, BY and ACROSS conditions.
It efficiently pre-computes all cell specifications using the lazy operations of the Polars library.
The :class:`.Task` is an iterable where each member is an instance of a :class:`.Cell`.
A :class:`.Cell` contains all instances of :math:`a`, :math:`b`, and :math:`x` that satisfy the specified
conditions for a particular value.

.. code-block:: python

   from fastabx import Task

   task = Task(dataset, on="#phone", by=["next-phone", "prev-phone", "speaker"])

   print(len(task))
   # 117927
   print(task[0])
   # Cell(
   #         ON(#phone_ax = AO, #phone_b = IH)
   #         BY(next-phone_abx = NG)
   #         BY(prev-phone_abx = L)
   #         BY(speaker_abx = 6295)
   # )

To control the size and number of cells, a :class:`.Task` can be instantiated with an additional
:class:`.Subsampler`. The :class:`.Subsampler` implements the two subsampling methods done in Libri-Light.
First, it can cap the number of :math:`a`, :math:`b` and :math:`x` independently in each cell (with :code:`max_size_group`).
Second, when ACROSS conditions are specified, it can limit the number of distinct values
that :math:`x` can take for the ON attribute (with :code:`max_x_across`).

.. code-block:: python

   from fastabx import Subsampler, Task

   task = Task(dataset, on="#phone", by=["next-phone", "prev-phone"], across=["speaker"])
   print(len(task))
   # 5437695

   subsampler = Subsampler(max_size_group=10, max_x_across=5)
   task = Task(
	dataset,
	on="#phone",
	by=["next-phone", "prev-phone"],
	across=["speaker"],
	subsampler=subsampler,
   )
   print(len(task))
   # 1346484

Once the task is built, the actual evaluation is conducted using the :class:`.Score` class.
A :class:`.Score` is instantiated with the :class:`.Task` and the name of a distance (such as "angular", "euclidean", etc.).
After the scores of each :class:`.Cell` have been computed, they can be aggregated using the :meth:`.collapse` method.
The user can either obtain a final score by weighting according to cell size (using :code:`weigted=True`),
or they can aggregate by averaging across subsequent attributes (with :code:`levels=...`).

.. code-block:: python

   from fastabx import Score

   score = Score(task, "angular")
   abx_error_rate = score.collapse(levels=[("prev-phone", "next-phone"), "speaker"])
   print(abx_error_rate)
   # 0.033783210627340875

CLI
===

This package also provides a command line interface, a simple wrapper that exposes the :func:`.zerospeech_abx` function.


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

This library aims to be as clear and minimal as possible to make its maintenance easy, and the code readable and
quick to understand. It should be easy to incorporate different components into one's personal code, and not just
use it as a black box.

At the same time, it must be as fast as possible to calculate the ABX, both in forming triplets and calculating the
distances themselves, while offering the possibility to use any configuration of ON, BY, and ACROSS conditions.

The idea of creating yet again a new ABX library comes from the realization that the
`polars library <https://github.com/pola-rs/polars>`_ efficiently and easily solves the difficulties associated with
creating triplets.

We can write the creation of the triplets as some "join" and "select" operations on dataframes, then some "filter"
for subsampling. With polars, the full query is built lazily and then processed end-to-end. The backend will run several
optimizations for us, and can even run on GPU. We don't have to worry anymore about how to built the triplets in a clever manner.

The computation of the distances is similar as what is done in `Libri-Light <https://github.com/facebookresearch/libri-light/tree/main/eval>`_
and `ZeroSpeech 2021 <https://github.com/zerospeech/libri-light-abx2>`_. The distances functions have been modified to be
more memory efficient by avoiding large broadcastings. The important change is that now the DTW is computed with a
PyTorch C++ extension, with CPU (using OpenMP) and CUDA backends. The speedup is most noticeable on large cells,
such as those obtained when running the Phoneme ABX without context conditions.

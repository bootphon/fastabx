==========
User guide
==========

| If you are coming from ABXpy or Libri-light / ZeroSpeech 2021 ABX, see :ref:`this page <other libs>`.
| If you want to reproduce experiments from the paper, see: https://github.com/mxmpl/fastabx-paper.

Quickstart
==========

Simple example where the representations are points drawn from two Gaussians, and the category
to discriminate is which Gaussian a point came from:

.. code-block:: python

   import numpy as np
   from fastabx import Dataset, Score, Task

   rng = np.random.default_rng(0)
   features = np.concatenate([rng.normal(0, 1, (50, 8)), rng.normal(2, 1, (50, 8))])
   labels = {"phone": ["a"] * 50 + ["b"] * 50, "speaker": ["s1", "s2"] * 50}

   dataset = Dataset.from_numpy(features, labels)  # What to compare
   task = Task(dataset, on="phone", by=["speaker"])  # Which triplets to build
   score = Score(task, "euclidean")  # How to compare them

   print(score.collapse(levels=["speaker"]))
   # 0.03059999644756317

- :class:`.Dataset` holds the labels and gives access to the representations.
- :class:`.Task` turns the ON, BY and ACROSS conditions into :class:`.Cell` objects. ``on="phone"`` asks for
  the discriminability of one phone against another, ``by=["speaker"]`` compares only within a speaker.
- :class:`.Score` computes the ABX of every cell, and :meth:`.collapse` averages them into a single number.

That number is an **ABX error rate**, not an accuracy: lower is better, chance level is 0.5, and the 3% above
means the two Gaussians are well separated.

To run this on speech instead, keep the `Task` and `Score` lines and build the dataset from an item file and a
directory of features.

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

When the labels and the features both live in a single table (a CSV file, or a polars or pandas DataFrame),
use :meth:`.Dataset.from_dataframe` and point ``feature_columns`` at the columns holding the features.

.. code-block:: python

   import polars as pl
   from fastabx import Dataset

   df = pl.DataFrame(
       {
           "label": ["a", "a", "b", "b"],
           "x": [0.1, 0.2, 1.0, 1.1],
           "y": [0.0, 0.1, 0.9, 1.2],
       }
   )
   dataset = Dataset.from_dataframe(df, feature_columns=["x", "y"])

The ABX :class:`.Task` is built given a :class:`.Dataset` and the ON, BY and ACROSS conditions.
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
The user can either obtain a final score by weighting according to cell size (using :code:`weighted=True`),
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

.. argparse::
   :module: fastabx.__main__
   :func: build_parser
   :prog: fastabx

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

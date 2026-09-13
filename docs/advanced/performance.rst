.. _performance:

======================
Performance and memory
======================

fastabx keeps everything in memory, on a single device. This page describes what is allocated, when, and
which knob to turn when it does not fit.

Where the data lives
====================

A :class:`.Dataset` loads all the features at once, into a single tensor on one device: CUDA if a GPU is
visible, CPU otherwise. Every constructor takes a ``device`` argument to choose another one.

.. code-block:: python

   from fastabx import Dataset

   dataset = Dataset.from_item(item, features, 50)                  # CUDA if available
   dataset = Dataset.from_item(item, features, 50, device="cpu")    # forced on CPU
   dataset = Dataset.from_item(item, features, 50, device="cuda:1")  # second GPU

The tensor holds every token, concatenated along time: its size is
``total number of frames × dimension × 4 bytes`` in float32. A corpus of a few hundred
thousand triphones, at 50 Hz and 768 dimensions, corresponds to a few gigabytes.
Ten times that is too probably too large for your GPU, and the fix is either ``device="cpu"``, or pooling (see below).

One thing to add: the ``"angular"`` and ``"cosine"`` distances L2-normalize the dataset **in place** and append a
singularity column, so the tensor is rewritten one dimension wider.

While scoring
=============

Cells are not scored one by one. Cells that share the same X and A sets are gathered into a group and scored
together, so each pair of sequences is compared once instead of once per cell it appears in. Within a group,
the cost of a comparison is a frame-level lattice: comparing ``n`` sequences of at most ``s`` frames
against ``m`` sequences of at most ``t`` frames allocates ``n × m × s × t`` floats, which an
:ref:`alignment <alignment>` then reduces to one distance per pair.

That product is may lead to running out of memory. It is bounded by two environment variables described in
:ref:`perf-env`:

- :code:`FASTABX_MAX_SCORE_CHUNK_ROWS` caps how many sequences are compared against the group's X at once.
  Lower it first on an out-of-memory error.
- :code:`FASTABX_GATHER_CHUNK_ROWS` caps how many rows are gathered and padded in one read.

What dominates the runtime
==========================

Three quantities, in decreasing order of how often they matter:

**The number of triplets.** It grows with the product of the cell sizes, so a handful of large cells can cost
more than thousands of small ones. This is what the :class:`.Subsampler` is for:

.. code-block:: python

   from fastabx import Subsampler, Task

   subsampler = Subsampler(max_size_group=10, max_x_across=5)
   task = Task(dataset, on="#phone", by=["prev-phone", "next-phone"], across=["speaker"], subsampler=subsampler)

**The sequence lengths.** The lattice is quadratic in them, and dynamic time warping then runs over it.

**The alignment.** DTW is an efficient C++ extension but it is still work, which can be skipped via pooling.

Pooling
=======

:func:`.pool_dataset` collapses each token to a single vector, so every sequence has one frame:

.. code-block:: python

   from fastabx import pool_dataset

   pooled = pool_dataset(dataset, "mean")

When it is acceptable for your evaluation, it is the single biggest speedup available.

Progress and measurement
========================

Both long phases show a progress bar: building the dataset, and scoring the cells. Pass ``progress=False`` to
the :class:`.Dataset` constructors, to :class:`.Score`, or to :func:`.zerospeech_abx` to silence them in a
pipeline; the CLI takes ``--quiet``. Setting :code:`TQDM_DISABLE` hides every bar whatever the arguments say.

Building the :class:`.Score` is where the computation happens; it runs eagerly in the constructor.
:meth:`.Score.collapse` and :meth:`.Score.details` only aggregate numbers that already exist.

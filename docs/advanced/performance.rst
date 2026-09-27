.. _performance:

======================
Performance and memory
======================

fastabx keeps the dataset in memory on a single device. The large scoring intermediates are split into chunks
of bounded size. This page describes what is allocated, when, and which knob to turn when it does not fit.

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
``total number of frames × dimension × element size`` (4 bytes in float32, 8 in float64). Constructors preserve
the source dtype unless ``dtype=...`` requests a conversion. A corpus of a few hundred
thousand triphones, at 50 Hz and 768 dimensions, corresponds to a few gigabytes.
Ten times that is probably too large for your GPU: the fix is either ``device="cpu"``, or pooling (see below).

One thing to add: the ``"angular"`` and ``"cosine"`` distances L2-normalize the dataset **in place** and append a
singularity column, so the tensor is rewritten one dimension wider.

While scoring
=============

Cells are not scored one by one. Cells that share the same X and A sets are gathered into a group and scored
together, so each pair of sequences is compared once instead of once per cell it appears in. A group with
``nx`` X, ``na`` A and ``nb`` B in total goes through three steps.

**Gathering.** The rows of many small groups are read and padded in a single call to the accessor, up to
:code:`FASTABX_GATHER_CHUNK_ROWS` rows. A larger group reads its X at once, and its A and B chunk by chunk
when they are compared.

**Distances.** Comparing ``n`` sequences of at most ``s`` frames against ``m`` sequences of at most ``t``
frames builds a frame-level lattice of ``n × m × s × t`` distances, which an :ref:`alignment <alignment>` then
reduces to one distance per pair. The comparisons are split, on both the X side and the target side, so that
no lattice exceeds :code:`FASTABX_MAX_LATTICE_ELEMENTS` elements nor :code:`FASTABX_MAX_SCORE_CHUNK_ROWS`
target rows. Lower the first on an out-of-memory error. The alignment may allocate buffers of its own, of the
same order as the lattice.

**Counting.** The ABX decisions are counted without building the ``nx × na × nb`` triplets: each row of X-to-A
distances is sorted once, and every B is located in it by binary search. With :class:`.Constraints`, the
triplets are compared in chunks of 2\ :sup:`24`. The constraints are evaluated once per distinct combination of
the labels they use, not once per triplet.

What is not split: the dataset itself; the X of a group, gathered at once; and the ``nx × (na + nb)`` matrix of
sequence distances of a group, which is proportional to the number of pairs rather than of triplets.
:code:`FASTABX_REDUCTION_FLUSH_COLS` caps how many per-B counts are kept before they are reduced to cell scores.

Counts and numerical precision
==============================

Cell sizes and constrained denominators use Int64. The win/tie counts are exact Int64 integers too: a win counts
2 and a tie 1, and the total is halved only when the float64 cell score is computed. Scores exported in the
``Score.cells`` DataFrame are float64 too. This keeps large counts valid, but it does not make the final displayed
error rate an arbitrary-precision number.

Custom distances and alignments may return a different floating dtype from the input features. Scoring retains
that output dtype even when the target rows are chunked. A custom implementation must return a consistent dtype
and device across calls, and compute each pair independently of the other pairs in its batch.

.. _angular-numerics:

Angular distance and zero frames
================================

``"angular"`` and its alias ``"cosine"`` use the angle divided by pi for nonzero frames. Zero frames have no
direction, so fastabx uses an explicit convention: zero/zero has distance 0, and zero/nonzero has distance 1.
The rule applies before alignment, including to zero frames inside a sequence. Padding is still excluded by
the real sequence lengths passed to the alignment.

Normalization first scales each nonzero row by its largest absolute component, then computes its L2 norm.
This avoids overflowing or underflowing the norm for finite extreme inputs. Float16 and bfloat16 norm
accumulation uses float32; the normalized features retain the input dtype. Values already rounded to zero or
infinity before reaching fastabx cannot be recovered. Floating-point rounding still applies to distances and
can affect decisions near ties.

For layout compatibility, normalization continues to append one column: a tiny constant for nonzero rows,
zero for zero rows. Zero rows remain entirely zero, and angular distance handles them explicitly. Custom
accessors must preserve this zero-row invariant when implementing ``normalize_``.

.. warning::

   This corrects historical zero-frame behavior: earlier versions replaced zero frames with the positive
   uniform direction, creating an arbitrary directional preference. Scores involving zero frames can change.
   Stable normalization can also change decisions near numerical ties or with extreme feature magnitudes.
   There is no legacy-normalization switch; pin the exact previous package version and dependencies when
   reproducing historical results, and rebuild datasets from original features when upgrading.

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

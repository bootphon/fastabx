=============
API reference
=============

.. autofunction:: fastabx.zerospeech_abx

Standard classes and functions
==============================

Dataset
-------

.. autoclass:: fastabx.Dataset
   :members: labels, accessor, normalize_, from_dataframe, from_item, from_item_with_times, from_item_and_units, from_numpy

.. autoclass:: fastabx.InMemoryAccessor

.. autoclass:: fastabx.Batch


.. _reserved-labels:

.. note::
   **Reserved label names.** Any column of ``Dataset.labels`` can be used as an ON, BY or ACROSS condition,
   with one restriction: it cannot be named ``index``, ``score``, ``size``, ``is_valid``, ``__cell``,
   ``__group``, ``__lookup``, ``__pos`` or ``__triplet``, and it cannot end with ``_a``, ``_b`` or ``_x``.
   Those names are used internally when building and scoring the cells. Passing such a column to a
   :class:`.Task` raises a ``ValueError``: rename it beforehand. Columns not used as conditions are unaffected.

Task
----

.. autoclass:: fastabx.Task
   :members: cells, from_cells

Subsample
---------

.. autoclass:: fastabx.Subsampler

Score
-----

.. autoclass:: fastabx.Score
   :members: cells, collapse, details, write_csv

   See :ref:`alignment` to change how sequences spanning several frames are compared.

Advanced
========

Pooling
-------

Pooling collapses the frame-level features of each token into a single vector, so that every token is
represented by one fixed-size embedding instead of a variable-length sequence. This is useful when you
want token-level (rather than frame-level) representations: the comparison no longer needs an
:ref:`alignment <alignment>`, which makes the distance computation faster. Two methods are available:
``"mean"`` averages the frames, and ``"hamming"`` averages them using a Hamming window (giving less
weight to the boundary frames).

.. autofunction:: fastabx.pool_dataset

.. autoclass:: fastabx.PooledDataset

.. py:class:: fastabx.PoolingName
   :canonical: fastabx.pooling.PoolingName

   Type alias for ``Literal["mean", "hamming"]``.

Cell
----

.. autoclass:: fastabx.Cell
   :members: num_triplets, needs_alignment

Accessor
--------

.. autoclass:: fastabx.Accessor()
   :members: lengths, batched, normalize_, __len__, __getitem__, __iter__


Distance
--------

.. autofunction:: fastabx.abx_on_cell

.. py:class:: fastabx.DistanceName
   :canonical: fastabx.distance.DistanceName

   Type alias for ``Literal["euclidean", "cosine", "angular", "kl_symmetric", "identical"]``.
   ``"cosine"`` is an alias for ``"angular"``.

.. py:class:: fastabx.Distance
   :canonical: fastabx.distance.Distance

   Type alias for ``Callable[[torch.Tensor, torch.Tensor], torch.Tensor]``: a function taking two batches
   of representations and returning their pairwise **frame-level** distances, as a ``(n1, n2, s1, s2)``
   cost lattice. Reducing that lattice to one distance per pair of sequences is the job of an
   :ref:`alignment <alignment>`.

   Anywhere a ``DistanceName`` is accepted (:class:`.Score` and :func:`.abx_on_cell`) a callable of this
   shape is accepted in its place. Only the built-in ``"angular"`` and ``"cosine"`` names normalize the
   dataset, so a custom distance is handed the features exactly as they are.

.. _alignment:

Alignment
---------

A :py:class:`.Distance` compares individual frames while an ``Alignment`` turns the resulting ``(n1, n2, s1, s2)``
cost lattice into the ``(n1, n2)`` distance between the sequences themselves. Dynamic time warping is the
default. When every sample has a single frame (a pooled dataset, see `Pooling`_), the alignment is bypassed and the
frame cost is used directly.

.. py:class:: fastabx.AlignmentName
   :canonical: fastabx.alignment.AlignmentName

   Type alias for ``Literal["dtw"]``, the only alignment available for now.

.. autoclass:: fastabx.Alignment()
   :members: __call__

   Anywhere an ``AlignmentName`` is accepted, a custom callable satisfying this protocol is accepted too.
   The lattice and the two length tensors are passed positionally, so an implementation may name them freely.
   Extension entry-point for alignments that fastabx does not ship, such as an edit distance:

   .. code-block:: python

      def edit(cost: Tensor, sx: Tensor, sy: Tensor, *, symmetric: bool) -> Tensor:
          ...  # your dynamic program over the lattice

      Score(task, "identical", alignment=edit)

Constraints
-----------

.. py:class:: fastabx.Constraints
   :canonical: fastabx.constraints.Constraints

   Type alias for ``Iterable[pl.Expr]``.

   See :ref:`constraints` to understand how to use them.

.. autofunction:: fastabx.constraints_all_different

Environment variables
=====================

Behaviour
---------

.. _librilight-bug:

- :code:`FASTABX_WITH_LIBRILIGHT_BUG`: If set to 1, changes the behaviour of :meth:`.Dataset.from_item` to
  match Libri-Light. Every feature will now be one frame shorter. This should be set only if you want
  to replicate previous results obtained with Libri-Light / ZeroSpeech 2021. See :ref:`slicing` for more details
  on how features are sliced.
- :code:`TQDM_DISABLE`: If set, every fastabx progress bar is hidden, overriding ``progress`` arguments and ``--quiet``
  flags.

.. _perf-env:

Performance tuning
------------------

The variables below bound the size of the intermediate tensors in the scoring engine.
Normal usage should not require changing them: lower them if the scoring runs out of memory, raise them if you
have memory to spare and the cells are small.

- :code:`FASTABX_MAX_SCORE_CHUNK_ROWS` (default 8192): Maximum number of rows compared at once when scoring a
  group of cells. Turn down if you have an out-of-memory error.
- :code:`FASTABX_GATHER_CHUNK_ROWS` (default 8192): Maximum number of rows gathered and padded in a single
  batched read from the :class:`.InMemoryAccessor`.
- :code:`FASTABX_REDUCTION_FLUSH_COLS` (default 262144): Number of accumulated columns after which the
  per-cell reduction is flushed. Larger values amortise the reduction over more cells, at the cost of
  keeping more intermediate counts around.

Exceptions
==========

Building a Dataset
------------------

.. autoexception:: fastabx.InvalidItemFileError
.. autoexception:: fastabx.FrequencyTypeError
.. autoexception:: fastabx.FeaturesSizeError
.. autoexception:: fastabx.EmptyFeaturesError
.. autoexception:: fastabx.EmptyDataPointsError
.. autoexception:: fastabx.EmptyDatasetError
.. autoexception:: fastabx.NonContiguousIndicesError
.. autoexception:: fastabx.NonFiniteError
.. autoexception:: fastabx.TimesArrayDimensionError
.. autoexception:: fastabx.TimesArrayFrontiersError

Building a Task
---------------

.. autoexception:: fastabx.DuplicateConditionsError
.. autoexception:: fastabx.EmptyTaskError
.. autoexception:: fastabx.InputTypeError
.. autoexception:: fastabx.LabelReservedNameError
.. autoexception:: fastabx.LabelSuffixError
.. autoexception:: fastabx.UnknownConditionError
.. autoexception:: fastabx.PrecomputedCellsError
.. autoexception:: fastabx.InvalidCellError

Scoring
-------

.. autoexception:: fastabx.CollapseError
.. autoexception:: fastabx.EmptyScoreError
.. autoexception:: fastabx.IdenticalDistanceDimensionError
.. autoexception:: fastabx.IncompatibleNormalizationError
.. autoexception:: fastabx.InvalidLevelsError
.. autoexception:: fastabx.NoConstraintsError
.. autoexception:: fastabx.PoolingNormalizedError

ZeroSpeech ABX
--------------

.. autoexception:: fastabx.InvalidSpeakerOrContextError
.. autoexception:: fastabx.MissingMaxXAcrossError

Configuration
-------------

.. autoexception:: fastabx.InvalidEnvironmentVariableError
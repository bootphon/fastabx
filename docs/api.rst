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

Advanced
========

Pooling
-------

Pooling collapses the frame-level features of each token into a single vector, so that every token is
represented by one fixed-size embedding instead of a variable-length sequence. This is useful when you
want token-level (rather than frame-level) representations: the comparison no longer relies on DTW, which
makes the distance computation faster. Two methods are available: ``"mean"`` averages the frames, and
``"hamming"`` averages them using a Hamming window (giving less weight to the boundary frames).

.. autofunction:: fastabx.pool_dataset

.. autoclass:: fastabx.PooledDataset

.. py:class:: fastabx.PoolingName
   :canonical: fastabx.pooling.PoolingName

   Type alias for ``Literal["mean", "hamming"]``.

Cell
----

.. autoclass:: fastabx.Cell
   :members: num_triplets, use_dtw

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
   of representations and returning their pairwise distances.

Constraints
-----------

.. py:class:: fastabx.Constraints
   :canonical: fastabx.constraints.Constraints

   Type alias for ``Iterable[pl.Expr]``.

   See :ref:`constraints` to understand how to use them.

.. autofunction:: fastabx.constraints_all_different

Environment variables
=====================

.. _librilight-bug:

- :code:`FASTABX_WITH_LIBRILIGHT_BUG`: If set to 1, changes the behaviour of :meth:`.Dataset.from_item` to
  match Libri-Light. Every feature will now be one frame shorter. This should be set only if you want
  to replicate previous results obtained with Libri-Light / ZeroSpeech 2021. See :ref:`slicing` for more details
  on how features are sliced.
- :code:`FASTABX_OUTPUT`: Controls the output format of the ``fastabx`` CLI. Defaults to a human-readable
  ``"ABX error rate: ..."`` line; set to ``json`` (or ``jsonl``) to emit a single JSON object containing
  the score and all CLI arguments instead.

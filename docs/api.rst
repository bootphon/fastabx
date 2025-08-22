=============
API reference
=============

.. autofunction:: fastabx.zerospeech_abx

Standard classes and functions
==============================

Dataset
-------

.. autoclass:: fastabx.Dataset
   :members: labels, accessor, from_csv, from_dataframe, from_item, from_item_with_times, from_item_and_units, from_numpy

Task
----

.. autoclass:: fastabx.Task

Subsample
---------

.. autoclass:: fastabx.Subsampler

Score
-----

.. autoclass:: fastabx.Score
   :members: collapse, details, write_csv

Pooling
-------

.. autofunction:: fastabx.pooling

Advanced
========

Cell
----

.. autoclass:: fastabx.cell.Cell
   :members: num_triplets, use_dtw

Distance
--------

.. autofunction:: fastabx.distance.distance_on_cell
.. autofunction:: fastabx.distance.abx_on_cell

DTW
---

.. autofunction:: fastabx.dtw.dtw
.. autofunction:: fastabx.dtw.dtw_batch

Constraints
-----------

.. py:type:: fastabx.constraints.Constraints

   Type alias for ``Iterable[pl.Expr | pl.Series | str]``.

   Should be a valid input to ``pl.DataFrame.filter``.
   See :ref:`constraints` to understand how to use them.

.. autofunction:: fastabx.constraints.constraints_all_different

Environment variables
=====================

.. _librilight-bug:

- :code:`FASTABX_WITH_LIBRILIGHT_BUG`: If set to 1, changes the behaviour of :meth:`.Dataset.from_item` to
  match Libri-Light. Every feature will now be one frame shorter. This should be set only if you want
  to replicate previous results obtained with Libri-Light / ZeroSpeech 2021. See :ref:`slicing` for more details
  on how features are sliced.

=============
API reference
=============

.. autofunction:: fastabx.zerospeech_abx

Standard classes and functions
==============================

Dataset
-------

.. autoclass:: fastabx.Dataset
   :members: from_csv, from_dataframe, from_item, from_numpy

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

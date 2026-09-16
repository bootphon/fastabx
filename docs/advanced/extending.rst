.. _extending:

=================
Extending fastabx
=================

The pipeline is meant to be taken apart, each of its pieces can be replaced without touching the others.
This page goes through the extension points, from the most to the least common.

Pooling: one vector per token
=============================

By default a token is a sequence of frames, and comparing two tokens requires an alignment.
:func:`.pool_dataset` collapses each token into a single vector instead:

.. code-block:: python

   from fastabx import Dataset, Score, Task, pool_dataset

   dataset = Dataset.from_item(item, features, 50)
   pooled = pool_dataset(dataset, "mean")  # or "hamming"

   task = Task(pooled, on="#phone", by=["speaker"])
   print(Score(task, "angular").collapse(levels=["speaker"]))

``"mean"`` averages the frames; ``"hamming"`` averages them under a Hamming window, giving less weight to the
boundary frames, which are the ones most contaminated by the neighbouring units. The result is a
:class:`.PooledDataset`, usable anywhere a :class:`.Dataset` is. Since every sequence now has a single frame,
the alignment is bypassed entirely, see :ref:`performance`.

Pool before scoring, as above. An ``"angular"`` :class:`.Score` normalizes its dataset in place, and pooling
one that has already been normalized is refused with a :exc:`.PoolingNormalizedError`.

Custom alignments
=================

A :class:`.Distance` compares individual frames and produces an ``(n1, n2, s1, s2)`` cost tensor. An
:class:`.Alignment` reduces it to one distance per pair of sequences. Dynamic time warping is the
only one shipped, but it is just a dynamic program over the lattice, and so is the edit distance for example:

.. code-block:: python

   from torch import Tensor
   from fastabx import Score

   def edit(cost: Tensor, sx: Tensor, sy: Tensor, *, symmetric: bool) -> Tensor:
       """Reduce a (n1, n2, s1, s2) lattice to the (n1, n2) distances between sequences."""
       ...  # your dynamic program

   score = Score(task, "identical", alignment=edit)

Two rules an implementation has to respect:

- **Normalize by the length of the alignment path.** The ABX decision compares a X-to-A distance against a
  X-to-B distance; an unnormalized distance would systematically favour the shorter pair.
- **Read only the** ``(sx[i], sy[j])`` **sub-block of each pair.** Everything beyond those lengths is padding.

Hand-built triplets
===================

When the triplets you want cannot be expressed as ON, BY and ACROSS conditions, build the cells yourself and
hand them to :meth:`.Task.from_cells`. The DataFrame needs five columns: ``index_a``, ``index_b`` and
``index_x``, each a list of row indices into ``Dataset.labels``, plus a ``header`` and a ``description``
string used when displaying the cell.

.. code-block:: python

   import polars as pl
   from fastabx import Score, Task

   cells = pl.DataFrame(
       {
           "header": ["a-b"],
           "description": ["ON(phone_ax = a, phone_b = b)"],
           "index_a": [[0, 2, 4]],
           "index_b": [[1, 3, 5]],
           "index_x": [[0, 2, 4]],
       }
   )
   task = Task.from_cells(dataset, cells, is_symmetric=True)
   print(Score(task, "euclidean").collapse(weighted=True))

``is_symmetric`` says whether A and X are the same set. When it is ``True``, ``index_a`` and ``index_x`` must
be equal row by row: scoring drops the diagonal of the distance matrix to avoid comparing a token with itself,
and that only makes sense if the two lists are the same, in the same order. Everything is checked up front,
and a violation raises :exc:`.PrecomputedCellsError`.

One limitation: such a task has no condition columns, so :meth:`.Score.collapse` needs ``weighted=True``
rather than ``levels``.

Constraints on triplets
=======================

Conditions operate at the level of cells. To filter *inside* a cell — excluding triplets where A and X come
from the same speaker, for instance — pass :class:`.Constraints` to the :class:`.Score`. They are polars
expressions over the labels of the three members of a triplet, suffixed with ``_a``, ``_b`` and ``_x``.
:ref:`constraints` covers this in full.

Scoring a single cell
=====================

:func:`.abx_on_cell` is the primitive underneath everything else: it takes one :class:`.Cell` and returns its
ABX error rate. Useful to build your own loop over cells.

.. code-block:: python

   from fastabx import Task, abx_on_cell

   task = Task(dataset, on="#phone", by=["speaker"])
   print(abx_on_cell(task[0], "euclidean"))

Unlike :class:`.Score`, it does not normalize anything: with the default ``"angular"`` distance the
features must already be L2-normalized, via :meth:`.Dataset.normalize_`, or the result is silently wrong.

Custom accessors
================

``Dataset.accessor`` is typed as the :class:`.Accessor` protocol. The implementation that ships with fastabx,
:class:`.InMemoryAccessor`, holds every feature in one tensor; anything satisfying the protocol can take its
place, for example a memory-mapped store, a lazy reader, or a decoder that reconstructs features on demand:

.. code-block:: python

   import torch

   from fastabx import Batch, Dataset

   class MyAccessor:
       device: torch.device
       is_normalized: bool

       def __len__(self) -> int: ...
       def __getitem__(self, i: int) -> torch.Tensor: ...
       def __iter__(self): ...
       def lengths(self, indices: list[int]): ...
       def batched(self, indices) -> Batch: ...
       def normalize_(self) -> None: ...

   dataset = Dataset(labels=labels, accessor=MyAccessor(...))

The scoring engine only ever reads through ``lengths`` and ``batched``, so those two are the ones that have to
be fast; ``batched`` is where a lazy implementation would do its I/O, gathering many indices at once. Indices
are the row numbers of ``Dataset.labels``: item ``i`` of the accessor describes row ``i``.

Custom distances
================

A :class:`.Distance` is any callable taking two batches of representations and returning their pairwise
frame-level distances as an ``(n1, n2, s1, s2)`` lattice. Like an alignment, it can be passed wherever the
name of a built-in distance is accepted, to :class:`.Score` and to :func:`.abx_on_cell`:

.. code-block:: python

   import torch
   from torch import Tensor
   from fastabx import Score

   def manhattan(a1: Tensor, a2: Tensor) -> Tensor:
       """Frame-level L1 distance, as a (n1, n2, s1, s2) lattice."""
       n1, s1, d = a1.size()
       n2, s2, _ = a2.size()
       lattice = torch.cdist(a1.view(n1 * s1, d), a2.view(n2 * s2, d), p=1)
       return lattice.view(n1, s1, n2, s2).transpose(1, 2)

   score = Score(task, manhattan)

The two batches are ``(n, s, d)`` tensors, padded to a common length ``s``; the alignment that consumes the
lattice is the one that knows the real lengths and reads only the valid sub-block of each pair. Only the
built-in ``"angular"`` and ``"cosine"`` names L2-normalize the dataset, so a custom distance receives the
features exactly as they are. If a custom distance requires normalization, perform it inside the callable, without
mutating its inputs. Functions, callable objects and callable dataclass instances are accepted.
``Score`` rejects custom distances on a dataset already normalized with :meth:`.Dataset.normalize_`,
because that normalization also appended a singularity-border feature and the original representation is gone.
Build a fresh ``Dataset``/``Task`` for the custom metric.

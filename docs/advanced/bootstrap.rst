.. _bootstrap:

=========
Bootstrap
=========

An ABX score is an estimate, and like any estimate it comes with sampling error. :py:class:`.Bootstrap`
quantifies that error by rescoring the task on many with-replacement resamples of the items, giving you
a confidence interval alongside the point estimate.

Basic usage
===========

Pass a :py:class:`.Bootstrap` to :py:class:`.Score`:

.. code-block:: python

   from fastabx import Bootstrap, Dataset, Score, Task

   dataset = Dataset.from_item("data.item", "features/", frequency=50)
   task = Task(dataset, on="phone", by=["speaker"])
   score = Score(task, "angular", bootstrap=Bootstrap(200, seed=0))

   print(score.collapse(levels=["phone", "speaker"]))               # the point estimate
   print(score.confidence_interval(levels=["phone", "speaker"]))    # (lower, upper), 95% by default

:py:meth:`.Score.bootstrap_collapse` gives you the whole distribution as a NumPy array, collapsed with
exactly the same logic as :py:meth:`.Score.collapse`, so you can compute whatever summary you want:

.. code-block:: python

   replicates = score.bootstrap_collapse(levels=["phone", "speaker"])
   print(replicates.std())

What gets resampled
===================

Each replicate redraws every item pool with replacement, keeping its size. A "pool" is a group of items
sharing the same ``on``/``by``/``across`` labels — exactly the groups the cells are built from. This is a
*stratified* bootstrap: the design of the task is held fixed and only the tokens inside it are resampled.

Pools are resampled **once per replicate, not once per cell**. Cells that share a pool (the A pool of the
``(a, b)`` cell is the B pool of the ``(b, a)`` cell, and every ``(a, *)`` cell of a given speaker shares
its A pool) therefore see the same resampled tokens within a replicate. That is what makes the replicate a
coherent "what if I had collected a different sample of tokens" scenario, rather than an independent
perturbation per cell that would ignore the correlation between cells built on the same tokens.

Because a resampled pool is represented as *multiplicities over the unique items* rather than as a list
with repeats, all the replicates of a cell share a single distance computation. Adding 100 replicates to a
run typically costs well under twice the point estimate alone, since the DTW and distance work — the
expensive part — is done once.

.. note::
   Reusing the same ``Bootstrap`` instance across two ``Score`` calls resamples both identically, because
   the draws are cached on the pool contents. That is what you want when comparing two models or two
   distances: take the difference of their bootstrap distributions replicate by replicate, and the
   comparison is paired.

Caveats
=======

**Do not bootstrap a subsampled task.** With a :py:class:`.Subsampler`, a cell only holds
``max_size_group`` items, so resampling them measures the variability *of the subsample*, not of the data.
Build the ``Task`` without a subsampler when bootstrapping.

**Tiny cells give degenerate replicates.** A cell with only two A items draws the same item twice half the
time, leaving no valid ``x != a`` pair. Those replicates are scored ``None`` and skipped by the collapse,
in the same way constrained cells with no valid triplet are. Tasks made of very small cells — typically the
"across" condition — therefore need more replicates to reach a given precision.

**This measures token sampling only.** The resampling happens *within* the existing ``on``/``by``/``across``
groups, so the interval reflects the variability of the tokens, with the set of speakers, contexts and
phones held fixed. If you want to generalise to new speakers, you need to resample speakers, which this
does not do.

**Constraints are not supported yet.** :py:class:`.Constraints` and ``bootstrap`` cannot be combined: the
per-triplet mask is indexed by position within the cell, while the bootstrap reweights the items behind
those positions.

Memory
======

The cached draws take ``4 * n_replicates * (total pool size)`` bytes and the per-replicate cell scores take
``8 * num_cells * n_replicates`` bytes. A task with 100 000 cells and 200 replicates therefore holds about
160 MB of replicate scores. Reduce ``n_replicates`` if that is too much.

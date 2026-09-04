.. _bootstrap:

=========
Bootstrap
=========

An ABX score is an estimate, and like any estimate it comes with sampling error. The bootstrap quantifies
that error by collapsing the task on many resamples of the data, giving you a distribution, and so a
confidence interval, alongside the point estimate.

Two things can be resampled, and they answer different questions:

- **Tokens**, the items inside each cell: *how much would the score move with other recordings of the same
  speakers?* The speakers, contexts and phones are held fixed.
- **Categories**, the values of a condition such as the speakers: *how much would the score move with other
  speakers?*

Basic usage
===========

Resampling speakers needs no extra scoring. Pass the conditions to resample to
:py:func:`fastabx.stats.bootstrap_collapse`:

.. code-block:: python

   from fastabx import Dataset, Score, Task
   from fastabx.stats import bootstrap_collapse, percentile_interval

   dataset = Dataset.from_item("data.item", "features/", frequency=50)
   task = Task(dataset, on="phone", by=["speaker"])
   score = Score(task, "angular")
   levels = [("phone", "phone_b"), "speaker"]

   replicates = bootstrap_collapse(score, levels=levels, resample=["speaker"], n_replicates=1000, seed=0)
   print(score.collapse(levels=levels))          # the point estimate
   print(percentile_interval(replicates))        # (lower, upper), 95% by default

Resampling tokens needs the distances, so it is done while scoring: pass a :py:class:`.Bootstrap` to
:py:class:`.Score`, then collapse the replicates the same way.

.. code-block:: python

   from fastabx import Bootstrap

   score = Score(task, "angular", bootstrap=Bootstrap(200, seed=0))
   replicates = bootstrap_collapse(score, levels=levels)

In both cases each replicate is collapsed with exactly the same logic as :py:meth:`.Score.collapse`, and
``levels``/``weighted`` have the same meaning. The result is a NumPy array, so you can compute whatever
summary you want (``replicates.std()``, ...). A replicate left with no valid cell is ``nan``.

Which one should I use?
=======================

Resample the **highest random level only**. Usually that means the speakers.

The token variability is already part of the spread between speakers: each speaker's score is computed from
its own tokens. Resampling speakers therefore already accounts for the tokens, and resampling both counts
the token variability twice and overestimates the variance. ``bootstrap_collapse`` allows it, but emits a
:py:class:`.NestedBootstrapWarning`.

Resampling tokens alone is the right choice when the speakers are fixed by design, for instance to compare
two models on a fixed benchmark.

.. note::
   If the statistic you care about is a mean over speakers (and :py:meth:`.Score.collapse` is), its variance
   can also be computed exactly, without any resampling. The bootstrap is most useful for statistics that are
   not means: medians, maxima, ratios, rankings, or correlations with behavioural data.

What gets resampled
===================

Categories
----------

A category drawn ``k`` times counts as ``k`` distinct categories with identical cells, and a category that was
not drawn drops out of the average. Nothing is rescored: only the weights of the collapse change.

Every column of a condition shares the same draw. A cell involving two values of the same condition, such as
the ``phone``/``phone_b`` contrast, or the ``speaker``/``speaker_x`` pair of an across task, weighs
``w(a) * w(b)``. A value is never paired with one of its own copies, so with very few phones, a replicate that
draws a single phone several times has no contrast left and is ``nan``.

The draws only depend on ``seed`` and on the name of the condition. Two scores collapsed with the same seed
see the same speakers in each replicate: the difference of their bootstrap distributions, replicate by
replicate, is a paired comparison.

Resampling several *crossed* conditions at once, such as speakers and contexts, overestimates the variance,
because their interaction gets counted more than once.

Tokens
------

Each replicate redraws every item pool with replacement, keeping its size. A "pool" is a group of items
sharing the same ``on``/``by``/``across`` labels: exactly the groups the cells are built from. This is a
*stratified* bootstrap: the design of the task is held fixed and only the tokens inside it are resampled.

Pools are resampled **once per replicate, not once per cell**. Cells that share a pool (the A pool of the
``(a, b)`` cell is the B pool of the ``(b, a)`` cell, and every ``(a, *)`` cell of a given speaker shares
its A pool) therefore see the same resampled tokens within a replicate. That is what makes the replicate a
coherent "what if I had collected a different sample of tokens" scenario, rather than an independent
perturbation per cell that would ignore the correlation between cells built on the same tokens.

A resampled pool is represented as *multiplicities over the unique items* rather than as a list with
repeats, so all the replicates of a cell share a single distance computation. When the alignment dominates
the cost (variable-length items with DTW), adding replicates is cheap. With pooled or single-frame features,
the distance is cheap and the replicates dominate instead: expect a cost that grows with ``n_replicates``.

The draw of a pool only depends on the seed and on the pool's contents. Two :py:class:`.Score` built with
the same ``Bootstrap`` seed and number of replicates resample their shared pools identically, which makes
comparisons paired.

Caveats
=======

**A subsampled task cannot be bootstrapped.** With a :py:class:`.Subsampler`, a cell only holds
``max_size_group`` items, so resampling them would measure the variability *of the subsample*, not of the data.
:py:class:`.Score` raises a :py:class:`.BootstrapSubsampledTaskError`. Category resampling with a ``Score``
built without a ``Bootstrap`` is not affected.

**Tiny cells give degenerate token replicates.** A cell with only two A items draws the same item twice half
the time, leaving no valid ``x != a`` pair. Those replicates are ``nan`` and skipped by the collapse, in the
same way constrained cells with no valid triplet are. Tasks made of very small cells, typically the "across"
condition, therefore need more replicates to reach a given precision.

**Constraints are not supported with token resampling.** :py:class:`.Constraints` and ``bootstrap`` cannot be
combined: the per-triplet mask is indexed by position within the cell, while the bootstrap reweights the items
behind those positions. Category resampling works with constraints.

Memory
======

Token resampling caches its draws in ``4 * n_replicates * (total pool size)`` bytes, and stores the
per-replicate cell scores and sizes in ``16 * num_cells * n_replicates`` bytes. A task with 100 000 cells and
200 replicates therefore holds about 320 MB of replicates. Reduce ``n_replicates`` if that is too much.

The collapse processes the replicates in chunks bounded by ``FASTABX_BOOTSTRAP_CHUNK_ELEMS`` (see
:ref:`perf-env`).

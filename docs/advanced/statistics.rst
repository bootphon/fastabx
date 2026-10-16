.. _statistics:

==========
Statistics
==========

An ABX error rate averages many speakers, and often many trained models. Both are samples: other speakers, or
another training seed, would give another number. ``fastabx.stats`` estimates how much, in closed form, and
compares groups of models the way Schatz et al. (2021) do in their *Supplementary Materials and Methods 4*.

fastabx only provides what is specific to ABX, or not found in standard libraries. The tests themselves, the
p-values and the corrections for multiple comparisons are left to SciPy, as shown in the recipes below.

The variance of a crossed design
================================

:py:func:`fastabx.stats.crossed_mean` takes a long table holding one value for every combination of some
*factors*, such as every speaker x model, and returns its mean and the unbiased estimator of the variance of
that mean:

.. code-block:: python

   from fastabx.stats import crossed_mean

   crossed_mean(table, factors=["speaker", "model"], value="score", by=["phone", "phone_b"])
   # -> phone, phone_b, estimate, variance, n_speaker, n_model

The levels of each factor are treated as independent samples of a population: speakers drawn from all
speakers, models from all the models the training procedure could produce. The mean is then a generalized
U-statistic, and the variance accounts for every source at once: speakers, models, and their interactions.
``by`` gives independent groups, each with its own mean and variance.

- With a single factor, the variance is the usual ``var(ddof=1) / n``.
- A factor with a single level (one model) is treated as fixed: the variance is then conditional on that level.
  It is ``null`` when every factor has a single level.
- Being unbiased, the variance can be slightly negative when its true value is close to 0. ``np.sqrt`` then gives
  ``nan``.

The table must be *fully crossed*: one value for every combination of the factors, in each group. Otherwise
:py:class:`.InvalidDesignError` is raised.

Error bars over speakers
------------------------

:py:meth:`.Score.details` gives a table with one row per speaker, as long as ``speaker`` is not in the
``levels``. For one model:

.. code-block:: python

   per_speaker = score.details(levels=[("prev-phone", "next-phone"), ("phone", "phone_b")])
   crossed_mean(per_speaker, factors=["speaker"])
   # -> estimate, variance, n_speaker

The estimate is ``score.collapse(levels=[("prev-phone", "next-phone"), ("phone", "phone_b"), "speaker"])``, and
``sqrt(variance)`` its standard error over speakers. For several training seeds of the same model, stack their
tables first:

.. code-block:: python

   import polars as pl

   levels = [("prev-phone", "next-phone"), ("phone", "phone_b")]
   table = pl.concat([s.details(levels=levels).with_columns(model=pl.lit(i)) for i, s in enumerate(scores)])
   crossed_mean(table, factors=["speaker", "model"])

The unit must be a BY condition
-------------------------------

Each value must belong to a single level of each factor. This holds for a ``speaker`` BY condition, but not
when ``speaker`` is an ACROSS condition: every cell then involves two speakers, which are not independent units.
Use the :ref:`bootstrap <bootstrap>` for across-speaker tasks.

Comparing two groups of models
==============================

:py:func:`fastabx.stats.compare` implements the analysis of Schatz et al. Each model is collapsed with
:py:meth:`.Score.details`, which averages the ``levels`` within each speaker. With ``E1[u, m]`` and
``E2[u, m]`` the resulting error rates in the two groups, the estimate is the mean of ``E1[u, a] - E2[u, j]`` over
every speaker ``u``, model ``a`` of the first group and model ``j`` of the second. The variance is the one of
``crossed_mean`` with the factors speaker x model of group 1 x model of group 2.

.. code-block:: python

   from fastabx.stats import compare

   differences = compare(
       english_models,         # list of Score, e.g. 10 models trained on American English
       japanese_models,        # list of Score, e.g. 10 models trained on Japanese
       unit="speaker",
       levels=[("prev-phone", "next-phone")],
       symmetric=True,
   )
   # -> phone, phone_b, estimate, variance, n_speaker, n_model_1, n_model_2

- ``symmetric=True`` merges the ``(a, b)`` and ``(b, a)`` cells into one contrast, whose error rate is the mean
  of the two, as in the paper. A contrast is dropped where one of its directions does not exist, for instance
  where ``a`` has a single item (it cannot be both A and X).
- The conditions left after ``levels``, besides the unit, define one comparison per group: here, one per
  contrast. Averaging the contrast as well, with ``levels=[("prev-phone", "next-phone"), ("phone", "phone_b")]``,
  gives a single comparison over all contrasts.
- Every Score must have the same valid cells, typically by being scored on Tasks with the same labels and the
  same constraints, so that both groups average the same contexts within each speaker. Otherwise
  :py:class:`.MismatchedScoresError` is raised.
- Without ``levels``, nothing is averaged: there is one comparison per cell, the unit set aside.
- With one model per group, the variance is over speakers only, conditional on the two models: this is the paired
  comparison of two models on the same test set.

Testing
-------

``z = estimate / sqrt(variance)`` is asymptotically normal, which gives a two-sided p-value. When the variance is
negative (its unbiased estimate can fall below 0: rare with many speakers and models, common with few) or
``null`` (no factor with two levels), there is no valid test. Count the comparison as not significant rather than
dropping it, so that a correction for multiple comparisons still covers it:

.. code-block:: python

   import numpy as np
   import scipy.stats

   def two_sided_p(differences: pl.DataFrame) -> np.ndarray:
       """p-value of each comparison, 1 when its variance is negative or null."""
       z = (differences["estimate"] / differences["variance"].sqrt()).fill_nan(None)
       return 2 * scipy.stats.norm.sf(z.abs().fill_null(0).to_numpy())

   p_value = two_sided_p(differences)
   differences = differences.with_columns(
       p_value=p_value, p_adjusted=scipy.stats.false_discovery_control(p_value, method="by")
   )

Use the Benjamini-Yekutieli correction (``method="by"``) rather than Benjamini-Hochberg: comparisons that share
models and test speakers are correlated, and BY is valid under any dependence. Which comparisons to correct
together is up to you; the next section discusses the paper's case.

.. note::
   The paper's equation divides the estimate by the variance. This is a typo: the statistic divides by its
   square root, the standard deviation.

Robust predictions
------------------

Schatz et al. report a contrast as a *robust* prediction when, in each of the 8 combinations of training register
(for both languages) and test register:

1. the difference estimated with models trained on 1/10th of the data is significant after the BY correction,
   always in the same direction;
2. the difference estimated with the models trained on the full data has the same sign and a larger absolute
   value.

The paper does not say which comparisons the correction covers. Two readings are valid:

- **(a) Every contrast in every combination**, 8 x N tests for N contrasts, then require all 8 to be
  significant. This is probably the paper's procedure. It controls the false discovery rate of the 8 x N
  individual tests, and is the more conservative reading.
- **(b) One test per contrast.** Requiring significance in all 8 combinations is an *intersection-union test*:
  its null hypothesis is "no difference in at least one combination", and it keeps its level when each
  combination is tested at that level, with no correction across the 8 (Berger, 1982). The p-value of a contrast
  is then the largest of its 8 p-values, and the correction covers the N contrasts only. It controls the false
  discovery rate of the robust contrasts themselves, which is what the claim needs.

Both take a few lines of polars, with one ``compare`` per combination:

.. code-block:: python

   levels = [("prev-phone", "next-phone")]  # one comparison per contrast
   contrast = ["phone", "phone_b"]

   def compare_all(english, japanese):
       """Compare the models of each register combination (dicts: combination -> Scores)."""
       return pl.concat([
           compare(english[r], japanese[r], levels=levels, symmetric=True).with_columns(registers=pl.lit(r))
           for r in combinations
       ])

   tenth = compare_all(en_tenth, jp_tenth)
   tenth = tenth.with_columns(p_value=two_sided_p(tenth))
   tenth = tenth.with_columns(p_adjusted=scipy.stats.false_discovery_control(tenth["p_value"], method="by"))
   per_contrast = (
       tenth.join(compare_all(en_full, jp_full), on=[*contrast, "registers"], suffix="_full")
       .group_by(contrast)
       .agg(
           # tested in every combination, with both data sizes, and the criterion 2
           consistent=(pl.len() == len(combinations))
           & (pl.col("estimate").sign().n_unique() == 1)
           & (pl.col("estimate").sign() == pl.col("estimate_full").sign()).all()
           & (pl.col("estimate_full").abs() > pl.col("estimate").abs()).all(),
           significant=(pl.col("p_adjusted") < 0.05).all(),  # (a)
           # (b): 1 when a combination is missing, as the test needs all of them
           p_max=pl.when(pl.len() == len(combinations)).then(pl.col("p_value").max()).otherwise(1.0),
       )
   )
   # (a) one correction over every contrast and combination
   robust_a = per_contrast.filter(pl.col("consistent") & pl.col("significant"))
   # (b) one correction over the contrasts
   p_contrast = scipy.stats.false_discovery_control(per_contrast["p_max"], method="by")
   robust_b = per_contrast.filter(pl.col("consistent") & pl.Series(p_contrast < 0.05))

With a single full-data model per language, the full-data comparison is conditional on those two models, and only
its estimate is used, as in the paper.

Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics*, 24(4), 295-300.

Closed form or bootstrap?
=========================

For a mean over speakers, as every ``collapse`` is, the closed form is exact, deterministic and immediate: prefer
it. The bootstrap of a crossed design overestimates the variance, and resampling only the speakers misses the
variance due to model training. Use the :ref:`bootstrap <bootstrap>` for statistics that are not a mean (medians,
correlations with behavioural data, rankings) and for across-speaker tasks.

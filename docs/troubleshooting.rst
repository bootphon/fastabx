.. _troubleshooting:

===============
Troubleshooting
===============

Every error below is raised by fastabx itself. They are grouped by the moment they happen.

Building a Dataset
==================

Most problems appear here, when the item file and the features meet for the first time.

:exc:`.InvalidItemFileError` — *"File extension ... is not supported"*
   The item file must be a ``.item``, ``.csv``, ``.jsonl`` or ``.ndjson`` file, and its extension decides how
   it is parsed. Rename a space-separated table to ``.item``, a comma-separated one to ``.csv``.

:exc:`FileNotFoundError` — *"N files missing to build the Dataset"*
   None, or not all, of the files named in the ``#file`` column were found under ``root``. Two usual causes:
   the ``extension`` argument does not match your files (``.pt`` by default), or the ``#file`` values are not
   the paths of the feature files relative to ``root``, without their extension. A bare ``utt1`` will not match
   ``root/spk1/utt1.pt``; the value has to be ``spk1/utt1``.

:exc:`.FeaturesSizeError` — *"Input features length is not correct for file ..."*
   The slice asked for by a row of the item file goes past the end of the features of that file. Usually one of:

   - The **feature frequency is wrong**. A file of 10 s at a real 50 Hz has 500 frames; ask for 100 Hz and
     every token in the second half of the file is out of range.
   - The model produced **one frame too few**, which happens with unpadded convolutions when the token sits at
     the very end of the file. Pad the convolutions, or add a little silence at the end of the audio.

:exc:`.EmptyFeaturesError` — *"N empty entries found"*
   Some tokens are shorter than a single frame at the given frequency, so their slice is empty. Check the
   frequency first. If it is right and you do intend to evaluate units that short, remove those rows from the
   item file. :ref:`slicing` explains how the frontiers are computed.

:exc:`.NonFiniteError` — *"Non-finite values detected in features"*
   A feature file contains a ``NaN`` or an infinity. fastabx refuses it rather than propagating it into the
   distances, where a single ``NaN`` silently poisons every triplet it takes part in. Look at the model that
   produced that file.

:exc:`.EmptyDataPointsError` — *"N empty elements were found in the dataset"*
   Same idea, raised by the accessor: some datapoints span zero frames.

:exc:`.EmptyDatasetError` — *"The dataset is empty"*
   There is no datapoint at all, so there is no task to build. Usually an item file whose rows were all
   filtered out upstream, or ``labels`` of length 0.

:exc:`.NonContiguousIndicesError` — *"The keys of ``indices`` must be exactly the row numbers ..."*
   From a hand-built :class:`.InMemoryAccessor`. Its ``indices`` must map every row of ``Dataset.labels``,
   from ``0`` to ``len(labels) - 1``, to its ``[start, end[`` frontiers. A gap would make the missing rows
   read as empty features rather than raise, so it is rejected up front.

:exc:`.FrequencyTypeError` — *"``frequency`` is getting converted to Decimal"*
   ``frequency`` was a ``float``. Pass an ``int`` (``50``), or a ``str`` for a non-integer frequency
   (``"12.5"``), so that the times stay exact.

:exc:`.TimesArrayDimensionError`, :exc:`.TimesArrayFrontiersError`
   From :meth:`.Dataset.from_item_with_times`: the times array of a file is not 1D, or no timestamp of a file
   falls between the ``onset`` and the ``offset`` of one of its tokens. In the second case, check that the
   times are in seconds and cover the whole file.

Building a Task
===============

:exc:`.LabelReservedNameError`, :exc:`.LabelSuffixError`
   A condition column uses a name fastabx needs internally — ``index``, ``score``, ``size``, ``is_valid``,
   ``__cell``, ``__group``, ``__lookup``, ``__pos``, ``__triplet`` — or ends with ``_a``, ``_b`` or ``_x``.
   Rename the column (``speaker_x`` → ``x_speaker``). See :ref:`the reserved names <reserved-labels>`.

:exc:`.DuplicateConditionsError`
   The same column appears twice across ``on``, ``by`` and ``across``. Each condition plays exactly one role.

:exc:`.UnknownConditionError` — *"... is not a column of ``Dataset.labels``"*
   A condition names a column the labels do not have, almost always a typo. The message lists every column
   that is available. Watch out for the ZeroSpeech names in particular: the ON condition of the standard
   task is ``#phone``, with the leading ``#``, not ``phone``.

:exc:`.EmptyTaskError` — *"The task has no cell"*
   No triplet satisfies the conditions, so there would be nothing to score. A cell needs two different
   values of the ON condition among datapoints sharing the same BY values, with at least two instances
   available for A. The most common cause is a BY condition that is redundant with the ON condition — if
   every speaker utters a single phone, then ``on="#phone", by=["speaker"]`` can never pair two phones
   together. Otherwise, the corpus is too small for the conditions asked for.

:exc:`.InputTypeError`
   A condition is not a string, or a :class:`.Subsampler` got a non-integer ``seed``. The subsampler also
   rejects sizes below 2: a cell needs at least two items to compare.

:exc:`.PrecomputedCellsError`
   From :meth:`.Task.from_cells`. The message says which rule was broken: a missing column among ``header``,
   ``description``, ``index_a``, ``index_b``, ``index_x``; an index column that is not a list of integers; an
   empty index list; an index outside the dataset; or, for a symmetric task, a row where ``index_a`` and
   ``index_x`` differ, or one holding fewer than two rows in ``index_a`` — scoring a symmetric cell drops the
   diagonal, so a single A leaves no triplet at all. See :doc:`advanced/extending`.

:exc:`.InvalidCellError`
   A hand-built :class:`.Cell` whose A, B and X are not all 3D tensors, do not share a feature dimension, or
   whose sizes do not match the number of items.

Scoring
=======

:exc:`.IdenticalDistanceDimensionError`
   The ``"identical"`` distance compares discrete units, so it needs features of shape ``(length, 1)``, one
   integer per frame. Encode each unit as a single number — :meth:`.Dataset.from_item_and_units` does this
   for you — or pick a distance defined on vectors.

:exc:`.IncompatibleNormalizationError`
   You scored a :class:`.Task` with the ``"angular"`` (or ``"cosine"``) distance, then scored again with
   another distance. The first :class:`.Score` L2-normalized the dataset **in place** and appended a
   singularity column, so the features are no longer in their original space and the second score would be
   silently wrong. Build a fresh :class:`.Dataset` for the other distance.

:exc:`.CollapseError`
   Either ``levels`` and ``weighted=True`` were both given, or neither was, on cells that have more than the
   two ON columns left to average. The message lists the columns that are still there: pass them through
   ``levels`` in the order you want them averaged, or ask for ``weighted=True``.

:exc:`.EmptyScoreError` — *"Every cell has a null score"*
   Only with :class:`.Constraints`, and only when *not a single* cell kept a valid triplet, so there is
   nothing left to average. Loosen the constraints, or check that they name the labels you meant (remember
   the ``_a``, ``_b`` and ``_x`` suffixes). When only some cells are empty they are skipped by the averages
   and nothing is raised.

:exc:`.InvalidLevelsError`
   ``levels`` is not a list of strings or tuples of strings, repeats a column, or names a column that is not
   in the scored cells.

:exc:`.NoConstraintsError`
   The :class:`.Constraints` expressions do not reference any column of ``Dataset.labels``. Remember the
   suffixes: a constraint on the ``speaker`` label is written with ``speaker_a``, ``speaker_b`` and
   ``speaker_x``. See :ref:`constraints`.

:exc:`.PoolingNormalizedError`
   :func:`.pool_dataset` was given a :class:`.Dataset` that an ``"angular"`` (or ``"cosine"``)
   :class:`.Score` had already L2-normalized in place. Its features carry an extra singularity column and
   are no longer in their original space, so pooling them would average that column in and quietly change
   the measure. Pool first, then score or build a fresh :class:`.Dataset` to pool.

ZeroSpeech ABX
==============

:exc:`.MissingMaxXAcrossError`
   :func:`.zerospeech_abx` was called with ``speaker="across"`` but no ``max_x_across``. It has no default
   because leaving it out is rarely intended: pass ``5`` for the original ZeroSpeech behaviour, or ``None``
   explicitly to disable the subsampling.

:exc:`.InvalidSpeakerOrContextError`
   ``speaker`` must be ``"within"`` or ``"across"``, and ``context`` ``"within"`` or ``"any"``.

Results that look wrong
=======================

**The score is around 0.5.** Chance level. The representations do not separate the categories at all — or the
labels are shuffled with respect to the features. Check that the rows of ``labels`` line up with the rows of
``features``.

**The score is suspiciously good.** Make sure nothing leaks into the triplets: the same recording appearing as
both A and X, for instance. :ref:`constraints` is the tool for excluding those.

**The number is the complement of what you expected.** fastabx reports the ABX **error rate**, so 0.03 means
3% errors, i.e. 97% discriminability.

**A score is null.** Only happens with :class:`.Constraints`: the cell has no valid triplet left. Those cells
are dropped from the averages rather than counted as zero.

**Scores differ from Libri-Light or ZeroSpeech 2021.** Expected, and explained in :ref:`other libs`: their
slicing drops one frame per token. Set ``FASTABX_WITH_LIBRILIGHT_BUG=1`` to reproduce the old numbers.

Out of memory
=============

Both the features and the intermediate distances live on the same device. See :ref:`performance` for what is
allocated, and for the arguments and environment variables that bound it.

:exc:`.InvalidEnvironmentVariableError`
   One of the ``FASTABX_*`` tuning variables of :ref:`perf-env` is set to something that is not a positive
   integer. The message names the variable; unset it to go back to the default.

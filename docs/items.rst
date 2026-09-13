.. _items:

==========
Item files
==========

An item file describes what to compare: one row per token, with the file it comes from, its position in
that file, and every attribute you may want to use as an ON, BY or ACROSS condition. It is the input of
:meth:`.Dataset.from_item`, :meth:`.Dataset.from_item_with_times` and :meth:`.Dataset.from_item_and_units`.

.. note::
   Item files are only needed when the tokens are segments of longer files, which is the usual case in speech.
   If you already have one representation per token, skip this page: :meth:`.Dataset.from_numpy` and
   :meth:`.Dataset.from_dataframe` take the features and the labels directly.

Before writing your own, look at :ref:`item-downloads`: the item files of the ZeroSpeech challenges and of
several papers are distributed as is, and cover some of the standard phoneme and triphone evaluations.

Format
======

An item file is a table. The ``.item`` extension is the historical format of ABXpy and ZeroSpeech, a text
table with space-separated columns and a header line:

.. code-block:: text

   #file onset offset #phone prev-phone next-phone speaker
   6295-244435-0009 0.2925 0.4725 IH L NG 6295
   6295-244435-0009 0.3725 0.5325 NG IH K 6295
   6295-244435-0009 0.4325 0.5725 K NG AH 6295

Three extensions are accepted, and the extension alone decides how the file is read:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Extension
     - Read as
   * - ``.item``
     - Space-separated text table with a header line.
   * - ``.csv``
     - Comma-separated text table with a header line.
   * - ``.jsonl``, ``.ndjson``
     - Newline-delimited JSON, one object per token.

Anything else raises :exc:`.InvalidItemFileError`.

Columns
=======

Three columns are required. Their default names come from the ZeroSpeech item files, and each can be renamed
through the corresponding argument of the ``Dataset`` constructors:

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Column
     - Argument
     - Meaning
   * - ``#file``
     - ``file_col``
     - Which file the token belongs to. See `Matching the features`_ for how it is resolved.
   * - ``onset``
     - ``onset_col``
     - Start time of the token, in seconds.
   * - ``offset``
     - ``offset_col``
     - End time of the token, in seconds.

Every other column is a label, and any of them can be used as an ON, BY or ACROSS condition of a
:class:`.Task`. The item file above gives ``#phone``, ``prev-phone``, ``next-phone`` and ``speaker``, which is
what the standard triphone task needs, but the set is free: add ``dialect``, ``word``, ``language``, anything
your evaluation conditions on. A handful of column names are reserved, see :ref:`the reserved names <reserved-labels>`.

``onset`` and ``offset`` are read as exact decimals. This matters because those times are multiplied
by the feature frequency to find frame indices, and a rounding error there moves a frontier by one frame.

.. _matching-features:

Matching the features
=====================

:meth:`.Dataset.from_item` takes the item file and a ``root`` directory, and looks for every file under
``root`` whose name ends with ``extension`` (``.pt`` by default), recursively. A feature file is identified by
**its path relative to** ``root``\ **, without the extension**, and that is the string the ``#file`` column
must contain:

.. code-block:: text

   root/
   ├── 1272/
   │   └── 128104/
   │       └── 1272-128104-0000.pt   ->   #file = 1272/128104/1272-128104-0000
   └── 6295-244435-0009.pt           ->   #file = 6295-244435-0009

A mismatch raises a :exc:`FileNotFoundError` saying how many files were found out of how many were expected.
If your item file uses bare utterance ids but your features are in per-speaker subdirectories, flatten the
directory, or rewrite the column, whichever is easier.

Each matched file is passed to ``feature_maker``, which defaults to :func:`torch.load` and must return a 2D
tensor of shape ``(frames, dimension)``. Pass your own if you prefer to compute the representations on the fly
instead of loading them from disk, see the tutorial :doc:`examples/external` with the model called inside
``feature_maker``.

Frequency
=========

``frequency`` is the number of feature frames per second, and it is what converts the onset and offset times
into frame indices. It must be an ``int``, a ``str`` or a :class:`~decimal.Decimal`:

.. code-block:: python

   from fastabx import Dataset

   dataset = Dataset.from_item("./triphone-dev-clean.item", "./features", 50)      # 50 Hz, one frame / 20 ms
   dataset = Dataset.from_item("./triphone-dev-clean.item", "./features", "12.5")  # 12.5 Hz, as a string

:ref:`slicing` gives the exact rule used to turn ``[onset, offset]`` into a range of frames. When your features
come with their own array of timestamps, use :meth:`.Dataset.from_item_with_times` instead and drop
``frequency`` altogether. When your tokens are discrete units listed in a single JSONL file, use
:meth:`.Dataset.from_item_and_units`.

The ZeroSpeech item files
=========================

:func:`.zerospeech_abx` and the ``fastabx`` CLI build their conditions themselves, so they expect the columns
of the ZeroSpeech item files to be present and named exactly: ``#phone``, ``prev-phone``, ``next-phone`` and
``speaker``, next to the three required ones. With any other naming, build the :class:`.Task` yourself.

.. _item-downloads:

Download an item file
=====================

.. list-table::
   :widths: 25 30 25 20
   :header-rows: 1

   * - Dataset
     - Language
     - Kind
     - Download URL
   * - `DiscoPhon <https://benchmarks.cognitive-ml.fr/discophon>`_
     - 12 languages, see the `paper <https://arxiv.org/abs/2603.18612>`__.
     - Triphone and phoneme
     - `benchmark data <https://cognitive-ml.fr/downloads/phoneme-discovery/discophon_data.tar.gz>`__
   * - ZeroSpeech 2021
     - LibriSpeech dev-clean, dev-other, test-clean, test-other
     - Triphone
     - `item files <https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz>`__
   * - ZeroSpeech 2021, phoneme ABX
     - LibriSpeech dev-clean, dev-other, test-clean, test-other
     - Phoneme
     - `item files <https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-phoneme.tar.gz>`__

If you released an item file that is not listed here, please open an issue or a pull request on
`the fastabx repository <https://github.com/bootphon/fastabx>`_ so that it can be added.

Writing your own
================

Nothing ties item files to speech. Any table with a file, a start and an end is usable: frames of a video,
windows of a time series, spans of a token stream. The example below is a complete, valid item file for a
corpus of two files with three tokens each, where the categories are shapes and colors:

.. code-block:: text

   #file onset offset shape color
   scene1 0.00 0.50 circle red
   scene1 0.50 1.00 square blue
   scene1 1.00 1.50 circle blue
   scene2 0.00 0.50 square red
   scene2 0.50 1.00 circle red
   scene2 1.00 1.50 square blue

.. code-block:: python

   from fastabx import Dataset, Score, Task

   dataset = Dataset.from_item("./shapes.item", "./features", 10)
   task = Task(dataset, on="shape", by=["color"])
   print(Score(task, "euclidean").collapse(levels=["color"]))

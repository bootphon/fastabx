.. _other libs:

============================
Coming from other libraries
============================

ABXpy
=====

`ABXpy <https://github.com/bootphon/ABXpy>`_ is the first implementation of ABX.
It has been used for the ZeroSpeech 2015, 2017 and 2019 challenges.
It uses numpy to compute distances and HDF5 format to store features and cells.
The part of the code involving HDF5 (using the tables, h5py, and h5features libraries) is very slow
and only works with some very specific version of tables and h5py.
fastabx has a similar high level API, but it is much faster.

If you have pre-computed features stored in the h5features format, and you want to use fastabx with those,
you can convert them to PyTorch tensors using the script located at :code:`scripts/convert_features.py`.
ABXpy stores features alongside "times" arrays. If you want to use the times array to build your :class:`.Dataset`
instead of directly computing the slicing indices, use the :meth:`.Dataset.from_item_with_times` class method.
See :ref:`slicing` for more details on how features are sliced. 

ABXpy has some additional features that do no exist in fastabx, such as other subsampling methods, Levenshtein distance, etc.
Please file an issue if you would like to see some of those features implemented in fastabx.

Libri-Light / ZeroSpeech 2021
=============================

The second implementation of ABX comes from `Libri-Light <https://github.com/facebookresearch/libri-light/tree/main/eval>`_.
It has been integrated into the ZeroSpeech 2021 challenge, and `extended here <https://github.com/zerospeech/libri-light-abx2>`_
to allow for controlling the context condition.

fastabx provides much more flexibility than Libri-Light, while being faster. If you only want the final triphone or phoneme
ABX error rate, you can use the :code:`fastabx` CLI or :func:`.zerospeech_abx` function that take arguments similar
as those provided to the Libri-Light codebase.

.. warning::
  There is a bug in the Libri-Light (and ZeroSpeech 2021) ABX codebase.
  When slicing the features to find the frames between the "onset" and "offset" of the phoneme / triphone,
  the end index is incorrect. The features have always one frame less than what they should.
  This is especially problematic for models that have large features (40 or 80 ms).
  The bug is fixed in fastabx, but if you want to still keep the Libri-Light behaviour, set
  :code:`FASTABX_WITH_LIBRILIGHT_BUG=1` (see :ref:`this <librilight-bug>` for reference).
  Overall, with the bug fixed the discriminability is a bit worse.
  See :ref:`slicing` for more details on how features are sliced.

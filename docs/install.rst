=======
Install
=======

Latest release
==============

Install the package with pip:

.. code-block:: console

   $ pip install fastabx

fastabx requires Python 3.12 or later, and depends on PyTorch 2.10.0 or later, NumPy, Polars, tqdm, and `torchdtw <https://github.com/bootphon/torchdtw>`_.

fastabx is available on Linux x86-64 (with glibc 2.34 or later [#glibc]_), macOS arm64, and Windows x86-64.

Build from source
=================

To build a wheel:

1. Clone the repository:

.. code-block:: console

   $ git clone https://github.com/bootphon/fastabx.git
   $ cd fastabx

2. Run the following command to build a wheel with ``uv``:

.. code-block:: console

   $ uv build --wheel

.. note::
   On Linux and macOS, make sure to have ``CXX=g++``. If you want to build
   with CUDA support, you must have the CUDA toolkit installed and set the
   ``CUDA_HOME`` environment variable. If you are on a cluster with a module
   system, you can probably load the ``cuda/12.4`` module.

Footnotes
---------

.. [#glibc] The glibc constraint is due to the runners available in GitHub CI that are used to build torchdtw.
   If you build from source, you can use the lowest version of glibc supported by PyTorch.

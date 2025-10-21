=======
Install
=======

Latest release
==============

Install the package with pip:

.. code-block:: console

   $ pip install fastabx

fastabx requires Python 3.12 or later, and PyTorch 2.9.0 [#torch]_ (with CUDA 12.8 on Linux).
The pinned PyTorch version is the default one on PyPI.
Wheels compatible with other versions and variants of PyTorch are available on the GitHub Releases page.

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

.. [#torch] The DTW relies on the C++ API of PyTorch which has no API/ABI compatibility between releases.
   This means you have to have to use the exact version of PyTorch this package was compiled against.

.. [#glibc] The glibc constraint is due to the runners available in GitHub CI. If you build from source,
   you can use the lowest version of glibc supported by PyTorch.

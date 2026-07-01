Installation
===============

Quick Install (End Users)
-------------------------

If you just want to use the ``regrender`` command (no cloning, no dev setup), install it as a
`uv tool <https://docs.astral.sh/uv/guides/tools/>`_ straight from GitHub:

.. code-block:: bash

    uv tool install git+https://github.com/ytsimon2004/regrender.git

This installs ``regrender`` into an isolated environment and puts it on your ``PATH``. Verify with:

.. code-block:: bash

    regrender --help

To upgrade later, or to remove it:

.. code-block:: bash

    uv tool upgrade regrender
    uv tool uninstall regrender

.. note::

    If ``regrender`` is not found after install, run ``uv tool update-shell`` (then restart your
    shell) to add uv's tool directory to your ``PATH``.

The sections below are for **developers** who want an editable checkout.

Prerequisites
-------------

* Python 3.12 or higher
* Git (for cloning the repository)

First, clone the repository and navigate to the project directory:

.. code-block:: bash

    git clone https://github.com/ytsimon2004/regrender.git
    cd regrender


UV Environment (Recommended)
-------------------------------

UV is a fast Python package installer and resolver. If you don't have UV installed, install it first:

.. seealso::

    https://docs.astral.sh/uv/getting-started/installation/

Then set up the environment:

.. code-block:: bash

    # Create virtual environment
    uv venv

    # Activate environment
    source .venv/bin/activate         # Linux/macOS
    .venv\Scripts\activate           # Windows

    # Install package in development mode
    uv pip install -e .

Conda Environment
------------------

If you prefer using Conda for environment management:

.. code-block:: bash

    # Create conda environment with Python 3.12
    conda create -n regrender python=3.12 -y

    # Activate environment
    conda activate regrender

    # Install package in development mode
    pip install -e .

Verification
------------

After installation, verify that regrender is working correctly:

.. code-block:: bash

    # Check if the command is available
    regrender --help

.. note::

    Atlas data is **not** downloaded by regrender directly. The Allen CCF atlas is fetched
    automatically by BrainGlobe/neuralib the first time you run ``register``, ``roi``, or
    ``probe`` (this may take a few minutes on first use).

.. note::

    If you encounter issues with ``llvmlite`` or ``numba`` dependencies on Python 3.12, this is a known compatibility issue with some atlas visualization libraries. The core registration functionality will still work.
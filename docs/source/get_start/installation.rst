Installation
===============

Three ways to install, depending on whether you just want the ``regrender`` command or an
editable developer checkout:

- **Option 1 — one-shot uv tool** (no clone, no env): just use the CLI. *Recommended for end users.*
- **Option 2 — uv virtual environment**: editable checkout for development.
- **Option 3 — conda environment**: editable checkout for development.

Prerequisites
-------------

* Python 3.12 or higher
* Git (for Options 2 and 3)

Option 1 — One-shot install with uv (no environment)
----------------------------------------------------

If you just want to run the ``regrender`` command, install it as a
`uv tool <https://docs.astral.sh/uv/guides/tools/>`_ straight from GitHub — no cloning, no
virtual environment to manage:

.. code-block:: bash

    uv tool install git+https://github.com/ytsimon2004/regrender.git

This installs ``regrender`` into an isolated environment and puts it on your ``PATH``. Upgrade or
remove it later with:

.. code-block:: bash

    uv tool upgrade regrender
    uv tool uninstall regrender

.. note::

    If ``regrender`` is not found after install, run ``uv tool update-shell`` (then restart your
    shell) to add uv's tool directory to your ``PATH``.

Option 2 — uv virtual environment (development)
-----------------------------------------------

For an editable checkout. `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ is a
fast Python package installer and resolver — install it first if you don't have it.

.. code-block:: bash

    # Clone
    git clone https://github.com/ytsimon2004/regrender.git
    cd regrender

    # Create and activate a virtual environment
    uv venv
    source .venv/bin/activate         # Linux/macOS
    .venv\Scripts\activate            # Windows

    # Install in editable mode
    uv pip install -e .

Option 3 — conda environment (development)
------------------------------------------

If you prefer conda for environment management:

.. code-block:: bash

    # Clone
    git clone https://github.com/ytsimon2004/regrender.git
    cd regrender

    # Create and activate a conda environment
    conda create -n regrender python=3.12 -y
    conda activate regrender

    # Install in editable mode
    pip install -e .

Verification
------------

After any of the above, verify that ``regrender`` is available:

.. code-block:: bash

    regrender --help

Download the atlas
------------------

regrender uses the Allen CCF atlas, which BrainGlobe/neuralib fetches automatically the first
time you run ``register``, ``roi``, or ``probe`` (this may take a few minutes). To pull it ahead
of time so the first real command doesn't stall:

.. code-block:: bash

    regrender setup

This fetches the 10 µm Allen mouse atlas (what regrender is built around). It's a no-op if the
atlas is already cached.

Data Directory Structure
=============================

regrender works on a **folder of serial section images** (``-D/--directory``) or a single
image (``-I/--image``). All outputs are written next to the input images. A typical folder
looks like this after running the full pipeline:

.. code-block:: text

    <slices_dir>/
        ├── <stem>.tif                       # input histology slices (one per section)
        │
        ├── transformations/                 # created by `regrender register`
        │     ├── <stem>_transform.json      # registration metadata (see below)
        │     ├── <stem>_transformed.tif     # histology warped into atlas space
        │     └── <stem>_overlay.png         # histology + boundaries, outside the brain transparent
        │
        ├── roi/                             # created by `regrender roi`
        │     ├── roi_points_raw.csv         # ROIs in raw pixel coords
        │     └── roi_points_ccf.csv         # ROIs projected into CCF (after "Project")
        │
        └── probe_shanks.csv                 # created by `regrender probe`

Registration metadata — ``<stem>_transform.json``
--------------------------------------------------

Written by ``regrender register`` and consumed by ``roi`` / ``probe``. Fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``matrix``
     - 3×3 homography (or affine) mapping the resized slice onto atlas-plane pixels.
   * - ``plane``
     - Cutting plane (``coronal`` / ``sagittal``).
   * - ``resolution``
     - Atlas resolution in µm (default ``10``).
   * - ``slice_index``
     - Atlas plane index (voxel) the slice was matched to.
   * - ``dw`` / ``dh``
     - Cutting-plane tilt offsets.
   * - ``rotate``
     - In-plane rotation (degrees) applied to the raw slice.
   * - ``flip_lr`` / ``flip_ud``
     - Whether the raw slice was flipped before registration.
   * - ``contrast``
     - ``(lo, hi)`` contrast window used for the saved ``.tif``, or ``null``.
   * - ``slice_xy`` / ``atlas_xy``
     - The matched landmark point pairs (slice pixels / atlas pixels).

The ``rotate`` / ``flip_lr`` / ``flip_ud`` fields record the preprocessing so raw ROI points
can be replayed into atlas space (raw → flip → rotate → resize → apply matrix).

ROI CSVs — ``roi/``
-------------------

``roi_points_raw.csv`` (from labeling on raw images):

- ``slice`` — source slice stem
- ``x``, ``y`` — ROI position in **raw image pixels**
- ``raw_h``, ``raw_w`` — raw image shape (needed to replay the transform)
- ``channel`` — ``merge`` / ``R`` / ``G`` / ``B``

``roi_points_ccf.csv`` (after "Project + Render" or ``roi --project``) adds:

- ``AP_location``, ``DV_location``, ``ML_location`` — bregma-relative CCF coordinates (mm)
- ``region`` — Allen region acronym at that point
- ``source`` — source slice stem
- ``channel``

Probe CSV — ``probe_shanks.csv``
--------------------------------

From ``regrender probe``:

- ``ap_mm``, ``dv_mm``, ``ml_mm`` — bregma-relative CCF coordinates (mm)
- ``probe_idx`` — shank index
- ``point`` — ``dorsal`` (superficial) or ``ventral`` (deep)
- ``source`` — source slice stem the point was picked on

The **Region profile plot** button also writes ``probe_region_profile.pdf`` and
``probe_region_profile.csv`` (one row per region span per shank: ``shank``, ``region``,
``extrapolated``, ``dv_start_mm``, ``dv_end_mm``, ``depth_start_mm``, ``depth_end_mm``,
``length_mm``).

.. note::

    Optional upstream preprocessing (channel split/merge, rescaling, ROI TIFF generation) can
    be done in Fiji/ImageJ — see the macros under ``res/fiji`` in the repository.

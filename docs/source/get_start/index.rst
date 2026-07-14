Getting Started
===============

Typical workflow
----------------

The commands are usually run in this order on a folder of serial sections (``-D <slices_dir>``):

1. ``regrender roi`` — label cells/ROIs on the raw slices (raw pixel coords, no registration needed yet).
2. ``regrender register`` — register each slice to the Allen CCF.
3. ``regrender roi --project --render`` — project the labelled ROIs into CCF space and render.
4. ``regrender probe`` — *(optional)* reconstruct probe/electrode tracks from dye labels on the
   registered slices.

ROIs are marked before registration but *projected* after it, because the saved transform is what
maps raw pixels into atlas space.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   register
   roi
   probe
   data_structure


Slice Registration (``regrender register``)
=============================================

Interactively register a histology slice to the Allen CCF in napari by matching landmark
point pairs between the atlas and your slice.

.. code-block:: bash

    # single image
    regrender register -I <image.tif>

    # folder of serial sections (step through with Prev/Next)
    regrender register -D <slices_dir>

Workflow
--------

The GUI shows the atlas plane on the left and your histology on the right.

1. Choose the cutting **plane** (coronal / sagittal), the **slice index** (atlas plane), and
   the ``dw`` / ``dh`` tilt offsets to match the atlas plane to your section.
2. Adjust the slice with **rotate** and **flip L-R / U-D** as needed.
3. Pick matched **landmark pairs** — click a point on the atlas, then the corresponding point
   on the slice (alternating). A homography (or ``--affine`` transform) is estimated from them.
4. Toggle **Preview** to overlay the warped histology under the atlas boundaries.
5. **Save** to write the registration to ``<output-dir>/`` (default ``<image-dir>/transformations``):

   - ``<name>_transform.json`` — matrix + metadata (see :doc:`data_structure`)
   - ``<name>_transformed.tif`` — histology warped into atlas space
   - ``<name>_overlay.tif`` — warped histology with atlas boundaries burned in

Use ``--load <…_transform.json>`` (with ``-I``) to resume a saved session — it restores the
points, slice index, tilt, rotation, and flips.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Meaning
   * - ``-I``, ``--image``
     - Histology image path (optional; can also load from the GUI).
   * - ``-D``, ``--directory``
     - Folder of serial sections; step through with Prev/Next.
   * - ``-P``, ``--plane-type``
     - Cutting orientation (``coronal`` / ``sagittal``; default ``coronal``).
   * - ``--resolution``
     - Atlas resolution in µm (default ``10``).
   * - ``-O``, ``--output-dir``
     - Output directory (default ``<image-dir>/transformations``).
   * - ``--name``
     - Output name (default: image stem).
   * - ``--flip-lr`` / ``--flip-ud``
     - Flip the histology before registration.
   * - ``--affine``
     - Use an affine instead of projective (homography) transform.
   * - ``--boundary-color``
     - Atlas boundary overlay color (matplotlib name or ``#hex``; default ``orange``).
   * - ``--load``
     - Resume from a saved ``*_transform.json`` (needs the matching ``-I`` image).

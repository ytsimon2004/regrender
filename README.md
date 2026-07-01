# regrender

**Tools for 2D mouse brain registration to the Allen Common Coordinate Framework (CCF)**

A napari-based Python toolkit
For registering histological brain slices to the Allen Brain Atlas, annotating cells/ROIs,
reconstructing probe tracks, and rendering the results with brainrender.

## Features

- **Interactive slice→CCF registration** (`regrender register`) — pick landmark pairs in napari, estimate a homography/affine transform.
- **ROI annotation** (`regrender roi`) — label cells on raw images, project them into CCF space, and render.
- **Probe-track reconstruction** (`regrender probe`) — reconstruct electrode shanks from dye labels across serial sections.
- **brainrender rendering** of ROIs and probes in 3D atlas space.

## Quick Start

```bash
# install (uv or pip)
uv pip install -e .

# 1. label ROIs on raw slices
regrender roi -D <slices_dir>

# 2. register each slice to the CCF
regrender register -D <slices_dir>

# 3. project ROIs to CCF + render  (or use the GUI "Project + Render" button)
regrender roi --project --render -D <slices_dir>

# probe tracks
regrender probe -D <slices_dir>
```

See the [full documentation](https://regrender.readthedocs.io) for the complete workflow.

## Contact

**Yu-Ting Wei** - ytsimon2004@gmail.com

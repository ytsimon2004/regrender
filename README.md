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

## Demo

**`regrender register`** — match landmark pairs between the atlas plane (left) and histology (right):

<img src="docs/source/_static/register.png" alt="register" width="800">

**`regrender roi`** — label cells per channel, then project + render in 3D with brainrender:

<img src="docs/source/_static/roi.png" alt="roi" width="800">

**`regrender probe`** — pick dye points per shank and reconstruct the tracks in 3D:

<img src="docs/source/_static/probe.png" alt="probe" width="800">


## Quick Start

```bash
# install as a CLI tool (no clone needed)
uv tool install git+https://github.com/ytsimon2004/regrender.git

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

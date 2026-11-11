# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-11-11

### Fixed
- Installation with napari 0.10: switched to the `napari[pyqt6]` extra, since 0.10 removed the `pyqt5` extra.
- Replaced the napari-deprecated `viewer.text_overlay` / `viewer.camera` usage in the register, ROI and probe apps.

### Changed
- Image folders are now listed in natural order (`slice_2` before `slice_10`) in the register, ROI and probe apps (#3, #4).
- Pinned `brainglobe-atlasapi>=2.3,<3`: 3.x moves atlases to a new cache layout, so existing `~/.brainglobe` atlases are not found.
- Declared `natsort>=8` as a direct dependency.
- Docs: `uv tool install` now uses `--python 3.12`, so uv does not pick an active conda or system Python older than 3.12.

### Added
- CI launch test that starts the apps against a synthetic offline atlas, with no GIN download needed.

## [0.1.1] - 2026-07-21

### Added
- GUI support for loading images or folders and for updating session data.

### Changed
- The preview resets when slice points change (rotation, ...).
- Folder mode skips unregistered slices with a warning, and raises an error if no slice is registered.

## [0.1.0]

- Initial release.

[0.1.2]: https://github.com/ytsimon2004/regrender/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ytsimon2004/regrender/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ytsimon2004/regrender/releases/tag/v0.1.0

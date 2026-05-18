# Changelog

All notable changes to this project will be documented in this file.

The project is currently in an experimental cleanup phase. Dates below describe repository/documentation milestones, not necessarily application releases.

## Unreleased

### Changed

- Repository direction changed from a commercial/internal README style to a public open-source project style.
- Added clearer repository hygiene rules.
- Added stronger `.gitignore` rules for local data, outputs, model weights, dependency caches and backup archives.
- Model weights are treated as local files and should not be synchronized with Git.
- `lista_filer.bat` should no longer contain private/local absolute paths.

### Added

- MIT license file.
- Public `README.md`.
- `requirements.txt` reflecting the current bootstrap dependency set.
- Documentation placeholders for architecture, development and repository hygiene.
- README placeholders for local-only folders such as `models/`, `indata/`, `sounds/`, `presets/` and `wheels/`.

### Removed / To remove from Git tracking

- Backup archives and old ZIP packages.
- Generated folder structure reports.
- Tracked model weights.
- Any private/local paths.

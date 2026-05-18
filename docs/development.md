# Development notes

## Working style

Use small, clear commits.

Recommended order:

1. repository hygiene,
2. documentation,
3. dependency cleanup,
4. code structure cleanup,
5. new features.

Do not combine major refactoring with functional changes.

## Current priorities

- Keep the public repository clean and understandable.
- Keep local data, outputs, archives and models out of Git.
- Keep `README.md` user-facing.
- Keep technical details in `docs/`.
- Keep `src/` changes for a later step.

## Coding conventions

- Keep module names clear and domain-based.
- Prefer small focused modules.
- Avoid hardcoded absolute paths.
- Use repository-relative paths.
- Avoid private local paths in scripts or documentation.
- Update documentation when structure changes.

## Testing checklist before release

Manual tests should include:

- app starts from `start.bat`,
- model picker works,
- video source picker works,
- line drawing works,
- zone drawing works,
- line AB/BA counts are recorded,
- zone IN/OUT counts are recorded,
- event CSV is created,
- heatmap OFF creates no heatmap files,
- heatmap ON writes only to `output/heatmap/`,
- snapshot-on-event works if enabled,
- sound test works,
- Abort works,
- ESC in preview aborts the run.

## Dependency notes

The current bootstrap launcher uses a local `_pkgs/` folder. This can be kept for now.

`requirements.txt` is added mainly for transparency and reproducibility. It reflects the dependency set used by the current bootstrap logic.

# Repository hygiene checklist

This repository should be source/documentation only.

## Should be committed

- source code,
- README and docs,
- small config files,
- license,
- dependency definitions,
- small placeholder README files for local folders.

## Should not be committed

- model weights,
- private input videos,
- generated outputs,
- backup ZIP files,
- old release packages,
- local dependency folders,
- cache folders,
- private/local paths,
- `.env` files,
- API keys or tokens,
- screenshots containing private data,
- large test datasets.

## Immediate cleanup actions

- [ ] Delete `Backups/` from the repository.
- [ ] Delete `folder_structure.txt` from the repository.
- [ ] Replace `lista_filer.bat` with the relative-path version.
- [ ] Remove tracked `.pt` model files from the repository.
- [ ] Keep only `models/README.md` and `models/.gitkeep` under `models/`.
- [ ] Replace the commercial/internal README with the public open-source README.
- [ ] Add `LICENSE`, `CHANGELOG.md`, `ROADMAP.md`, `requirements.txt`.
- [ ] Add `docs/`.
- [ ] Confirm that no local path such as `C:\Users\...` remains.

## Model weights

Model files should be stored locally:

```text
models/
```

They are ignored by Git.

Before committing, check that no file like this appears in GitHub Desktop:

```text
*.pt
*.pth
*.onnx
*.engine
*.tflite
```

## Input and output data

Input videos should be local:

```text
indata/
```

Outputs should be local:

```text
output/
```

Both are ignored by Git.

## Backups

Do not use GitHub as a backup folder.

Use Git history for project versions. Keep temporary backup files outside the repository.

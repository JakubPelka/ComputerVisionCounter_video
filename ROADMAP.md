# Roadmap

This roadmap focuses first on repository safety and maintainability, then on application improvements.

## Phase 1 — Repository hygiene

Status: in progress.

- [ ] Remove `Backups/` from the repository.
- [ ] Remove generated folder listings such as `folder_structure.txt`.
- [ ] Remove tracked model weights from Git.
- [ ] Keep `models/` local and documented with `models/README.md`.
- [ ] Replace private-path scripts with relative-path scripts.
- [ ] Replace commercial/internal README wording with public open-source wording.
- [ ] Add `LICENSE`, `requirements.txt`, `CHANGELOG.md` and basic docs.
- [ ] Verify that `.gitignore` prevents future accidental commits of outputs, archives, models and private files.

## Phase 2 — Documentation cleanup

- [ ] Keep one main `README.md`.
- [ ] Move technical details into `docs/architecture.md`.
- [ ] Add a short user guide in `docs/usage.md` if needed.
- [ ] Add a troubleshooting page for dependency, model, video and sound issues.
- [ ] Add screenshots later, only if they are generic and do not contain private data.

## Phase 3 — Code structure review

- [ ] Review large source files, especially the main processing runner.
- [ ] Split large files only when the current behavior is understood and preserved.
- [ ] Keep changes small and testable.
- [ ] Update architecture documentation after structural changes.

## Phase 4 — Functional verification

- [ ] Verify Quality slider and Advanced settings synchronization.
- [ ] Test line crossing counts with AB/BA direction.
- [ ] Test zone IN/OUT counts.
- [ ] Test event CSV schema.
- [ ] Test heatmap OFF mode: no heatmap files should be written.
- [ ] Test heatmap ON mode: files should be written only under `output/heatmap/`.
- [ ] Test snapshot-on-event behavior.
- [ ] Test sound alert modes.

## Phase 5 — Future feature ideas

Not for immediate cleanup.

- Live HUD with total detections even without drawn zones.
- More robust ID retention for difficult tracking situations.
- Peak concurrent object metrics.
- Dwell time metrics.
- Zone entry/exit timestamp summaries.
- Movement and location heatmaps.
- Better packaging for non-technical users.

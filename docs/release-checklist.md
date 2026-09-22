# Release Checklist

Use this checklist before pushing or publishing a larger batch of repository updates.

## Source code

- [ ] Add the frozen scientific drivers for the main thesis results.
- [ ] Add compact collectors and conversion scripts.
- [ ] Keep recovered helper code such as `t4_diag_common.py` with provenance notes.
- [ ] Remove generated caches and bytecode before committing.
- [ ] Confirm no private scratch paths are required for ordinary reading.

## Evidence and results

- [ ] Index every large archive in `docs/evidence-archive-index.md`.
- [ ] Keep large archives outside normal Git history.
- [ ] Record source hashes where available.
- [ ] Record job ids and scheduler evidence in evidence archives, not as loose logs.
- [ ] Ensure figure assets are either small curated images or external bundles.

## Documentation

- [ ] README explains the project, research questions, and documentation entry points.
- [ ] `docs/thesis-to-repository-map.md` maps chapters and claims to repo/evidence locations.
- [ ] `docs/experiment-status.md` reflects current status of RQ1--RQ5 and post-review controls.
- [ ] `docs/code-audit-status.md` reflects recovered files and remaining caveats.
- [ ] Terminology avoids overclaiming beyond the numerical evidence.

## Git hygiene

- [ ] `git status` is clean before final push.
- [ ] `git log --oneline origin/main..main` contains intentional commits only.
- [ ] No `.zip`, `.tar.gz`, `.vtu`, `.pvd`, `.h5`, `.msh`, or PBS output files are accidentally staged.
- [ ] Push only after confirming the batch is appropriate for the private GitHub repository.
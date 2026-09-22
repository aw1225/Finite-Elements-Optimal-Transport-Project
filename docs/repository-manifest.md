# Repository Manifest

This manifest describes the role of the top-level directories.

| Path | Role |
| --- | --- |
| `README.md` | Project overview and entry point. |
| `docs/` | Thesis mapping, audit status, evidence archive index, reproducibility guidance, and release checklist. |
| `src/` | Curated source code recovered from the thesis implementation archive. |
| `scripts/` | Small utilities for collection, conversion, and verification. |
| `experiments/` | Lightweight experiment definitions and parameter tables. |
| `jobs/` | Sanitized CX3/PBS job templates and submission notes. |
| `environment/` | Environment notes for Firedrake/PETSc/MPI execution. |
| `results/` | Compact Git-suitable summaries only. |

Large field outputs and archive bundles are intentionally referenced from documentation rather than stored directly in Git.
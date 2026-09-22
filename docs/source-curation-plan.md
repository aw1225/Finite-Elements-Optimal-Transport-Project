# Source Curation Plan

This page tracks which code families should be included before the repository is treated as a complete thesis companion.

## Include in source tree

| Family | Intended location | Status | Notes |
| --- | --- | --- | --- |
| Core heat-kernel Sinkhorn utilities | `src/ot_fas/core/` | pending | Shared log-domain Sinkhorn, heat-operator wrappers, gauge handling, and residual utilities. |
| Translation reference-validation drivers | `src/ot_fas/benchmarks/translation/` | pending | Include the source used for the discrete-reference story and any exact-error trajectory collectors. |
| Affine rotated-map drivers | `src/ot_fas/benchmarks/affine/` | pending | Include canonical affine/FAS drivers and small post-review robustness collectors. |
| Annulus and disk-to-annulus drivers | `src/ot_fas/benchmarks/annulus/` | pending | Include frozen annulus, T3, and associated map-export support where compact. |
| Frozen T1--T5 stress tests | `src/ot_fas/stress_tests/` | pending | Include the frozen production drivers, not exploratory duplicates. |
| Post-review T4 diagnostics | `src/ot_fas/postreview/t4/` | started | `t4_diag_common.py` has been recovered and added locally. |
| CSV/VTP map conversion utilities | `scripts/` or `src/ot_fas/visualization/` | pending | Include `csv_transport_to_vtp.py` and any small rendering helpers used for thesis figures. |
| Evidence collectors | `scripts/collectors/` | pending | Include compact collectors for m-check, W-check, m32 control, RQ1 matched refinement, and T4 supplemental diagnostics. |

## Exclude from source tree

- duplicate exploratory scripts not used in thesis evidence;
- generated `__pycache__` directories;
- PBS stdout/stderr logs;
- full raw result directories;
- VTK/VTP/PVD field payloads unless a release explicitly packages a small figure asset.

## Curation rule

A source file should be added only when at least one of the following is true:

- it is directly referenced by a thesis result;
- it is needed to reproduce a compact evidence collector;
- it explains a post-review diagnostic or visualization export;
- it is a stable shared utility used by the above.

Each added scientific driver should carry enough context to identify its corresponding archive, source hash, and thesis claim.
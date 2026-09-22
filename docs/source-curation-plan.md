# Source Curation Plan

This page tracks which code families should be included before the repository is treated as a complete thesis companion.

## Include in source tree

| Family | Intended location | Status | Notes |
| --- | --- | --- | --- |
| Core heat-kernel Sinkhorn utilities | `src/ot_fas/core/` | pending | Shared utilities remain embedded in historical drivers; extract only if needed. |
| Translation reference-validation drivers | `src/ot_fas/benchmarks/translation/` | copied | Includes v44 exact-error, v46 discrete-reference, and v51 four-level drivers. |
| Affine rotated-map drivers | `src/ot_fas/benchmarks/affine/` | partial | Post-review m-check, W-check, and m=32 controls copied; canonical v25/v26 affine scaling driver still to add if desired. |
| Annulus and disk-to-annulus drivers | `src/ot_fas/benchmarks/annulus/` | partial | Annulus v45 copied; T3 lives under stress-test drivers. |
| Frozen T1--T5 stress tests | `src/ot_fas/stress_tests/` | copied | T1--T5 drivers copied, with T4 v50 extracted from the complete suite archive. |
| Post-review T4 diagnostics | `src/ot_fas/postreview/t4/` | copied | Recovered helper plus supplemental and adjoint wrappers copied. |
| CSV/VTP map conversion utilities | `scripts/visualization/` | copied | `csv_transport_to_vtp.py` copied. |
| Evidence collectors | `scripts/collectors/` | partial | Core collectors copied for translation, annulus, affine controls, T3, T4, and T5; RQ1 matched-refinement collector can still be added. |

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
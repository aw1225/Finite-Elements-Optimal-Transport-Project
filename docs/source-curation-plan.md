# Source Curation Plan

This page tracks which code families should be included before the repository is treated as a complete thesis companion.

## Include in source tree

| Family | Intended location | Status | Notes |
| --- | --- | --- | --- |
| Core heat-kernel Sinkhorn utilities | `src/ot_fas/core/` | deferred | Shared utilities remain embedded in historical drivers; extract only if reuse or packaging becomes necessary. |
| Translation reference-validation drivers | `src/ot_fas/benchmarks/translation/` | copied | Includes v44 exact-error, v46 discrete-reference, and v51 four-level drivers. |
| Affine rotated-map drivers | `src/ot_fas/benchmarks/affine/` | copied | Canonical v25/v26 drivers plus post-review m-check, W-check, and m=32 controls copied. |
| Annulus and disk-to-annulus drivers | `src/ot_fas/benchmarks/annulus/` and `src/ot_fas/stress_tests/t3/` | copied | Annulus v45 and T3 disk-to-annulus drivers are copied. |
| Frozen T1--T5 stress tests | `src/ot_fas/stress_tests/` | copied | T1--T5 drivers copied, with T4 v50 extracted from the complete suite archive. |
| Post-review T4 diagnostics | `src/ot_fas/postreview/t4/` | copied | Recovered helper plus supplemental and adjoint wrappers copied. |
| CSV/VTP map conversion utilities | `scripts/visualization/` | copied | `csv_transport_to_vtp.py` copied. |
| Evidence collectors | `scripts/collectors/` and `results/provenance/` | copied/provenance-recorded | Core collectors copied for translation, annulus, affine controls, T3, T4, and T5. RQ1 matched-refinement was an inline collection pass; its PBS wrapper and compact job provenance are recorded instead of a standalone collector. |

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
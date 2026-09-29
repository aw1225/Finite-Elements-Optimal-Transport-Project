# Compact Result Summaries

This directory contains small, text-only summaries extracted from the verified evidence archives. It deliberately excludes raw solver output, full field files, mesh payloads, large compressed archives, and machine-local caches.

## Subdirectories

- `postreview-controls/`: matched robustness controls for heat-substep refinement, physical bridge-width checks, and the `m=32` FAS refinement-control run.
- `rq1-matched-refinement/`: fixed five-cycle RQ1 matched-refinement runs for annulus and translation, with job IDs and contraction summaries.
- `t4-supplemental/`: post-review T4 supplemental diagnostic summaries, adjoint probe summaries, and the mined safety-audit summary.
- `visualization/`: compact conversion summaries for recovered visualization payloads.

## Headline Files

- `headline_results.md`: short human-readable summary of the strongest post-review numerical controls.
- `t1_t5_safety_denominator_headlines.csv`: compact table of denominator-health headline values mined from the post-review safety audit.

For exact provenance, use these files together with `docs/evidence-archive-index.md`, `docs/reproducibility.md`, and `docs/curated-source-manifest.csv`.

# Evidence Archive Index

This page records the external evidence archives that support the thesis companion repository. Large archives are intentionally not committed to Git; the repository records their provenance, purpose, and expected local location.

## Post-review numerical controls

| Control | Local archive | Purpose |
| --- | --- | --- |
| Heat-substep endpoint check | `notes_FAS/affine_mcheck_evidence_2026-09-14.tar.gz` | Checks sensitivity of affine endpoint quantities to heat substep count `m`. |
| Periodic-width check | `notes_FAS/affine_wcheck_evidence_2026-09-14.tar.gz` | Checks robustness to periodic-domain width `W` under approximately fixed bridge spacing. |
| `m=32` FAS refinement control | `notes_FAS/affine_fas_m32_refinement_control_evidence_2026-09-14.tar.gz` | Tests whether refinement-insensitive FAS contraction observed at `m=16` persists at `m=32`. |
| Matched RQ1 refinement | `notes_FAS/rq1_matched_refinement_evidence_2026-09-14.tar.gz` | First-five-cycle matched refinement comparison for annulus and translation bridge pairs. Compact job provenance is recorded in `results/provenance/rq1_matched_refinement_jobs_2026-09-14.csv`. |
| T4 supplemental diagnostics | `notes_FAS/t4_supplemental_evidence_2026-09-16.tar.gz` | Post-review T4 reference, mechanism, adjoint, and state diagnostics. |

## Real-transport map exports

| Case | Local archive or directory | Notes |
| --- | --- | --- |
| T1 nonlinear crescent | `Stress Tests/T2/real_transport_csv_export_2026-09-16/t1_csv_export_4080955/` | Rank-split CSV map exports and derived VTP point clouds. |
| T2 split half-disks | `Stress Tests/T2/real_transport_csv_export_2026-09-16/t2_csv_export_4080956.tar.gz` | CSV export archive verified locally; VTP point-cloud conversion lives beside extracted archive. |
| T3 disk-to-annulus | `Stress Tests/T3/real_transport_csv_export_2026-09-17/t3_csv_export_4081445.tar.gz` | CSV export archive for the recovered T3 map fields. |
| T4 high-frequency square | `Stress Tests/T4/real_transport_csv_export_2026-09-16/t4_csv_export_4080957.tar.gz` | CSV export archive for T4 map and diagnostics. |
| Translation v46 FAS map | `notes_FAS/translation_csv_export_2026-09-18/translation_v46_fas_to_tol_vtp_bundle_4091708.zip` | Verified translation computed-map VTP bundle; not the self-debiased family. |
| Translation self-debiased safe/unsafe | `Stress Tests/translation_selfdebiased_safe_unsafe_images_bundle.zip` | Figure-ready safe/unsafe self-debiased translation images. |
| Annulus B460 ParaView bundle | `Stress Tests/annulus_B460_clean_fragile_folded_paraview_bundle.zip` | Figure-ready original annulus transition bundle. |

## Verification convention

For each archive, keep the accompanying `.sha256` or local `.local.sha256` file beside the archive. When unpacking an evidence archive, prefer internal `SHA256SUMS.txt` checks where present. If `SHA256SUMS.txt` includes itself, regenerate the checksum manifest excluding `SHA256SUMS.txt` before treating the internal manifest as canonical.

## Git policy

Do not commit these archives directly unless a release explicitly requires a small, curated asset. Git should contain:

- source files needed to understand the computation;
- compact CSV summaries when they are small and thesis-critical;
- provenance notes, hashes, and archive locations;
- scripts used to convert or inspect evidence.

Git should not contain raw rank-split field dumps, large ParaView payloads, PBS scratch directories, or uncompressed result trees.
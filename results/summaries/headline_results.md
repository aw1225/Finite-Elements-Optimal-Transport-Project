# Headline Numerical Summaries

Generated from the compact, verified evidence archives copied into this repository. Values below are intended as a navigation aid; the adjacent CSV/JSON files remain the authoritative machine-readable record.

## Post-Review Controls

| Control | Main observation | Authoritative files |
| --- | --- | --- |
| Heat-substep endpoint check | The affine endpoint remains target-tolerance clean for `m=8,16,32`; map error and Monge defect decrease as `m` is refined from 8 to 32. | `postreview-controls/affine_mcheck_v1_stopping_endpoints.csv`, `postreview-controls/affine_mcheck_v1_matched_updates.csv` |
| Physical bridge-width check | `W=3,4,5` runs all reached the production residual target with zero tangled area in the archived diagnostics. | `postreview-controls/affine_wcheck_comparison.csv` |
| `m=32` FAS refinement-control | Five-cycle contractions are closely matched across `B230->B460` and `B460->B920`: `rho_5=0.3985628602` and `rho_5=0.3986422328`. | `postreview-controls/affine_fas_m32_refinement_control_summary.csv` |
| Matched RQ1 refinement | Annulus and translation five-cycle contractions are refinement-insensitive across the tested bridge pair hierarchy. | `rq1-matched-refinement/rq1_matched_refinement_summary.csv` |
| T4 supplemental diagnostic | Frozen T4 reference was certified at 96 updates; one FAS cycle reduced residual and state error relative to the three-sweep smoother in the supplemental comparison. | `t4-supplemental/supplemental_summary.csv`, `t4-supplemental/supplemental_summary.json` |

## Matched RQ1 Five-Cycle Contractions

| Case | Pair | `rho_5` | Relative spread within case |
| --- | --- | ---: | ---: |
| Annulus | `B230_B460` | 0.3756779813 | 0.043% |
| Annulus | `B460_B920` | 0.3758380344 | 0.043% |
| Translation | `B230_B460` | 0.2184129017 | 0.090% |
| Translation | `B460_B920` | 0.2182156908 | 0.090% |

## T4 Supplemental Diagnostic Headline

| Quantity | Value |
| --- | ---: |
| Strict reference updates | 96 |
| Strict reference residual `p` | `6.841166030332677e-11` |
| Smooth-three residual `p` | `0.0020917020518061995` |
| One-cycle FAS residual `p` | `0.00021820277128275012` |
| FAS/smooth-three residual ratio | `0.10431828524255186` |
| FAS/smooth-three state-error ratio | `0.12543185534008172` |

## Notes

These are numerical evidence summaries, not theorem statements. The adjoint diagnostics are finite probe audits; they support the stated computational interpretation but do not prove an exact common discrete variational structure.

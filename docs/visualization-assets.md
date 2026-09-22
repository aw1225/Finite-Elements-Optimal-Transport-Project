# Visualization Assets

This repository does not store large ParaView payloads in normal Git history. Instead, generated field visualizations are archived as external evidence bundles. The files below are the current figure-ready assets produced from the verified local workspace.

## Original benchmark visuals

### Annulus

Primary original-case visual bundle:

```text
Stress Tests/annulus_B460_clean_fragile_folded_paraview_bundle.zip
```

Recommended files inside the bundle:

```text
annulus_B460_map_fields_clean_fragile_folded.pvd
annulus_B460_jacobian_fields_clean_fragile_folded.pvd
```

### Translation self-debiasing

Safe/unsafe self-debiased translation image bundle:

```text
Stress Tests/translation_selfdebiased_safe_unsafe_images_bundle.zip
```

Generated PNGs:

```text
safe_tau316_debiased_mesh.png
unsafe_tau237_debiased_mesh.png
safe_tau316_unsafe_tau237_overlay.png
source_safe_tau316_unsafe_tau237_triptych.png
```

Interpretation:

- `Tau316` is used as the safer self-debiased regime.
- `Tau237` is used as the unsafe or near-folding regime.

### Translation computed-map CSV/VTP export

The FAS translation computed-map export is useful as a verified map export, but it is not the self-debiased figure family:

```text
notes_FAS/translation_csv_export_2026-09-18/translation_v46_fas_to_tol_vtp_bundle_4091708.zip
```

Do not confuse this with the self-debiased Tau237/Tau316 visuals.

## T1--T4 real-transport map exports

The T1--T4 stress-test map fields were exported as rank-split CSV files and converted into VTP point-cloud files using:

```text
tools/csv_transport_to_vtp.py
```

The local generated VTP folders are:

```text
Stress Tests/T2/real_transport_csv_export_2026-09-16/t1_csv_export_4080955/vtp_point_clouds/
Stress Tests/T2/real_transport_csv_export_2026-09-16/t2_csv_export_4080956/vtp_point_clouds/
Stress Tests/T3/real_transport_csv_export_2026-09-17/t3_csv_export_4081445/vtp_point_clouds/
Stress Tests/T4/real_transport_csv_export_2026-09-16/t4_csv_export_4080957/vtp_point_clouds/
```

These VTP outputs are point-cloud representations of the exported finite-element degrees of freedom. They are intended for visual inspection and thesis figures, not as replacements for the archived numerical CSV evidence.
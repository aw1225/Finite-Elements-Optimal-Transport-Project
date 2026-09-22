# Code Audit Status

This page records which implementation files have been identified as the scientific authority for the thesis companion repository, and which remaining code-level gaps should be closed before public release.

## Authoritative implementation families

The thesis codebase is organized around finite-element heat-kernel Sinkhorn solvers and nonlinear Full Approximation Scheme (FAS) accelerators. The curated repository should treat the following families as the main authority:

- Translation and heat-operator refinement drivers.
- Affine rotated-map benchmark drivers.
- Annulus and disk-to-annulus robustness drivers.
- Frozen T1--T5 stress-test drivers.
- Post-review numerical-control collectors and wrappers.
- CSV-to-VTP visualization conversion scripts.

Large raw outputs, machine-specific PBS logs, temporary caches, and duplicate exploratory scripts are intentionally excluded from Git history unless they are needed to support a specific thesis claim.

## Verified post-review controls

The following post-review controls have archived evidence packages outside the normal Git history:

- Heat-substep endpoint robustness (`m`-check).
- Periodic-domain width robustness (`W`-check).
- Matched `m=32` FAS refinement-control.
- Matched RQ1 first-five-cycle refinement comparison.
- T4 supplemental diagnostics and adjoint checks.
- Real-transport map CSV/VTP exports for T1--T4.
- Translation self-debiased safe/unsafe ParaView figures.

These packages should be cited as evidence archives rather than copied into the repository wholesale.

## Known code-audit gap

One post-review T4 helper module remains missing from the local reading bundle:

```text
t4_diag_common.py
T4_diag_common.py
```

The frozen T4 v50 driver, supplemental wrapper logic, submitted job evidence, and output summaries are present. However, any functions imported from this missing helper, such as adjoint-audit or state-diagnostic helpers, should not be described as line-by-line audited until that file is recovered.

Recommended wording:

> The frozen T4 v50 implementation and submitted supplemental evidence are present and auditable. The only unresolved code-level gap is the helper module `t4_diag_common.py`, imported by the post-review diagnostics; its internal helper implementations were not present in the recovered reading bundle.

## Precision constraints for public documentation

Repository text should avoid overclaiming. In particular:

- distinguish the free-space Gaussian heat identity from its periodic finite-element realization;
- describe `p, q` as log-density scalings carefully;
- avoid implying that all coarse models are independently the same OT problem;
- distinguish FAS consistency from a convergence theorem;
- describe strict references as strict-criteria numerical references, not ground truth.
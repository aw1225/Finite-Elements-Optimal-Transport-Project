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

## Recovered T4 diagnostic helper

The post-review T4 helper module has been recovered and copied into the curated source tree:

```text
src/ot_fas/postreview/t4/t4_diag_common.py
```

This closes the previously recorded code-level gap for the T4 supplemental diagnostics. The helper provides the shared routines imported by the T4 wrappers, including frozen-source SHA checking, adjoint diagnostics, state diagnostics, barycentric-map diagnostics, and heat-count helpers.

Recommended wording:

> The frozen T4 v50 implementation, submitted supplemental evidence, and recovered helper module `t4_diag_common.py` are present in the curated repository. The helper imports and SHA-checks the frozen v50 source before running post-review diagnostics; it should be treated as diagnostic support code, not as a replacement for the frozen scientific driver.

## Precision constraints for public documentation

Repository text should avoid overclaiming. In particular:

- distinguish the free-space Gaussian heat identity from its periodic finite-element realization;
- describe `p, q` as log-density scalings carefully;
- avoid implying that all coarse models are independently the same OT problem;
- distinguish FAS consistency from a convergence theorem;
- describe strict references as strict-criteria numerical references, not ground truth.
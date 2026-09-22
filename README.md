# Efficient Multiscale Finite-Element Solvers for Entropic Optimal Transport Maps

This repository accompanies an MSc research project on finite-element
heat-kernel methods for entropic optimal transport and nonlinear multilevel
Full Approximation Scheme (FAS) acceleration.

The implementation applies log-domain Sinkhorn relaxation to a hierarchy of
finite-element approximations of the heat operator. The target regularisation
and transport state spaces remain fixed across the hierarchy, while recursive
coarse correction replaces some expensive fine-level heat applications with
cheaper coarse approximations.

## Research questions

The computational evidence is organised around five questions:

1. Does FAS convergence remain stable under refinement of the auxiliary
   heat-operator discretisation?
2. Does recursive multilevel correction reduce time to nonlinear tolerance,
   and do measured heat-operation costs explain the improvement?
3. Does gauge-invariant error against an independent fine discrete reference
   corroborate the residual convergence?
4. When does nested coarse-to-fine initialisation improve total solution time?
5. Does the frozen solver configuration transfer to more difficult transport
   geometries and spectral structures?

## Repository contents

The curated repository will contain:

- the verified finite-element Sinkhorn/FAS implementation;
- reproducible experiment definitions;
- sanitized batch-job templates;
- compact convergence and profiling summaries;
- independent reference-validation evidence;
- selected robustness benchmarks and figures;
- source hashes and provenance records.

Large field files, raw solver output, machine caches, temporary archives, and
institution-specific execution details are not stored in normal Git history.

## Companion documentation

- [Thesis-to-repository map](docs/thesis-to-repository-map.md)
- [Experiment status](docs/experiment-status.md)
- [Code audit status](docs/code-audit-status.md)
- [Visualization assets](docs/visualization-assets.md)
- [Evidence archive index](docs/evidence-archive-index.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Repository manifest](docs/repository-manifest.md)
- [Source curation plan](docs/source-curation-plan.md)
- [Release checklist](docs/release-checklist.md)

## Status

This repository is being assembled from a provenance-checked private research
archive. Experiments are added only after their source identity, execution
record, and numerical summaries have been verified.

## License and citation

License and citation metadata will be finalized before public release.

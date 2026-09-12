# Thesis-to-Repository Map

This repository follows the scientific structure of the thesis rather than
the chronological order in which exploratory code was written.

## Mathematical method

The reusable implementation will support the formulation developed in
Chapter 3:

- finite-element approximation of heat interactions;
- stabilized log-domain Sinkhorn iteration;
- gauge normalization of the dual state;
- a hierarchy of heat operators on a fixed transport state space;
- nonlinear FAS coarse correction;
- nested coarse-to-fine initialisation;
- residual, discrete-state-error, and map diagnostics;
- barycentric reconstruction and conditional spread.

## RQ1: Convergence under heat-operator refinement

Repository evidence:

- principal translation benchmark;
- selected bridge resolutions;
- fixed-cycle residual trajectories;
- refinement comparison summary.

## RQ2: Recursive multilevel efficiency

Repository evidence:

- fine-only Sinkhorn comparison;
- two-level and recursive multilevel FAS;
- level-wise heat-action counts;
- measured heat-operation costs;
- predicted and observed timing comparisons;
- three-level versus four-level hierarchy experiment.

## RQ3: Residual versus discrete solution error

Repository evidence:

- independently constructed fine discrete references;
- reference-certification traces;
- gauge-invariant state-error trajectories;
- residual and error contraction summaries.

## RQ4: Nested coarse-to-fine initialisation

Repository evidence:

- controlled initialisation protocol;
- preparation cost;
- subsequent fine-level work;
- total time-to-tolerance comparison.

## RQ5: Robustness beyond calibration problems

Repository evidence:

- T1 nonlinear crescent;
- T2 split support;
- T3 disk-to-annulus;
- T4 high-frequency structure;
- T5 thin-neck geometry.

Each robustness test will state whether it was frozen, confirmatory,
exploratory, amended, or subject to a reference limitation.

## Excluded material

The repository will not reproduce the entire private research workspace.
Intermediate debugging versions, repeated parameter searches, raw field
trees, caches, downloaded literature, and duplicate archives are excluded
unless they are required to support a specific thesis claim.

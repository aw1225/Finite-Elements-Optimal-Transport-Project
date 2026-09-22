# Reproducibility Guide

The thesis evidence has three reproducibility levels.

## Level 1: Documentation and provenance

This repository records the scientific sources, experiment families, evidence archive locations, and source hashes. This level is intended for readers who want to understand what was run and which archive supports each claim.

## Level 2: Collector and summary reproduction

For compact evidence archives, rerun the provided collectors against the archived result directories. This reproduces tables such as contraction factors, normalized residual spreads, heat-count summaries, and timing summaries without rerunning expensive finite-element solves.

## Level 3: Solver reruns on CX3

Full solver reruns require Firedrake, PETSc/MUMPS, MPI, and the Imperial CX3 module environment used by the thesis. The historic runs used PBS jobs with 4 MPI ranks and queue-specific memory requests. Reruns should preserve:

- the frozen scientific source file;
- the submitted PBS wrapper or a documented minimal wrapper patch;
- all environment variables controlling bridge resolution, heat substeps, stopping criteria, and map export;
- the source SHA256 recorded at runtime;
- the PBS job id and `qstat -xf` record.

## Map export convention

Native Firedrake VTK export was unreliable on CX3 for some stress-test runs. The stable route is:

1. use the existing map-writing branch in the frozen code;
2. replace Firedrake `VTKFile` with the CSV shim during the rerun;
3. archive rank-split CSV files with source and PBS provenance;
4. convert CSV fields to local VTP point-cloud files for ParaView/PyVista inspection.

This is a diagnostic export path. It should be described as retaining already-computed diagnostic fields, not as a new solver experiment.

## Terminology constraints

Use the same caution as the thesis text:

- `p, q` are log-density scalings;
- strict references are strict-criteria numerical references, not ground truth;
- FAS consistency is not a convergence theorem;
- coarse bridge models are cheaper approximations of the heat interaction, not independent replacements for the full optimal transport problem.
# Environment Notes

The thesis computations were run in a Firedrake/PETSc/MPI environment on CX3. Exact module availability is cluster-specific and may change over time.

Minimum conceptual dependencies:

- Python 3;
- Firedrake with PETSc support;
- mpi4py;
- MUMPS/Cholesky-capable linear solver stack;
- NumPy and standard-library CSV/JSON tooling;
- ParaView, PyVista, or VTK tooling for post-processing exported map fields.

For reproducibility, prefer archiving the module commands and `qstat -xf` output for each production run instead of relying on this high-level environment note alone.
# CX3 Job Templates

This directory is reserved for sanitized PBS templates and submission notes.

The original CX3 jobs used site-specific queues, module names, scratch paths, and user names. Public templates should therefore be explicit about what must be edited before reuse:

- queue name and memory class;
- Firedrake module version;
- project scratch path;
- number of MPI ranks;
- environment variables selecting the experiment and run id.

PBS output files, scheduler history dumps, and scratch result trees should not be committed here. Store those in evidence archives and cite them from `docs/evidence-archive-index.md`.
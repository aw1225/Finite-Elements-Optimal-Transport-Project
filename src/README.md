# Source Layout

The source tree is being curated from the private research archive.

## Current contents

- `ot_fas/postreview/t4/t4_diag_common.py`: recovered helper used by post-review T4 supplemental diagnostics.

## Intended contents

The public companion source tree should eventually include:

- core finite-element heat-kernel Sinkhorn drivers;
- FAS hierarchy drivers and support utilities;
- frozen T1--T5 stress-test drivers;
- compact collectors and CSV/VTP conversion utilities;
- provenance helpers for hashing and source identity checks.

Source files should be added with enough context to explain which thesis claim they support. Avoid adding exploratory duplicates unless they are needed to reconstruct a specific result.
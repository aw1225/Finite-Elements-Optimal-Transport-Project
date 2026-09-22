# T4 Post-Review Diagnostics

This directory contains recovered helper code for the post-review T4 supplemental diagnostics.

`t4_diag_common.py` was recovered after the initial repository audit. It provides shared routines used by the T4 supplemental diagnostic wrappers, including frozen-source SHA checking, adjoint diagnostics, state diagnostics, barycentric-map diagnostics, and heat-count helpers.

The helper does not replace the frozen T4 v50 scientific driver. It imports and verifies that driver before running supplemental diagnostics.
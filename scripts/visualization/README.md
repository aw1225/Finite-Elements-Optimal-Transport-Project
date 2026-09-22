# Visualization Utilities

Utilities for converting retained diagnostic map exports into visualization-friendly formats.

`csv_transport_to_vtp.py` converts rank-split CSV map fields into VTP point-cloud files for ParaView/PyVista inspection. This is used when native Firedrake VTK output is unreliable on CX3.
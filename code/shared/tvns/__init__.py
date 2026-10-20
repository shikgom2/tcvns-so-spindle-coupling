"""tVNS x SO-Spindle analysis pipeline (Python port of R_codes).

Modules
-------
config    path settings (E: -> D: switch)
io        data loaders (parquet / csv / pkl / fif)
stats     statistics (LMM, circular statistics, FDR, Hedges' g)
figtheme  Nature-style figures (port of figure_theme.R)
"""

__all__ = ["config", "io", "stats", "figtheme"]

# Prevent garbled Korean output under the default Windows console encoding (cp949).
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001  (pipes, redirects, etc.)
        pass

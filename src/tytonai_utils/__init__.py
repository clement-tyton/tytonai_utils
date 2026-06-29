"""tytonai_utils — reusable helpers for interacting with the tytonai platform.

Features (built incrementally):
1. Web map import   — download all tiles of a web map from a tile size + webmap link.
2. Manifest import  — download tiles listed in a manifest (image + mask .npz).
3. Model fetch      — download a model from its config file.
4. Mask rollup      — remap annotation mask categories.
"""

__version__ = "0.2.2"

# Force a non-interactive (Agg) matplotlib backend on import. The threaded downloads
# (e.g. download_grid) clash with GUI backends like Tk and can HARD-CRASH the interpreter
# (Tcl_AsyncDelete ... core dumped). Agg just renders to PNG. Honour an explicit user
# choice via the MPLBACKEND env var; do nothing if matplotlib isn't installed.
import os as _os

if "MPLBACKEND" not in _os.environ:
    try:
        import matplotlib as _matplotlib

        _matplotlib.use("Agg")
    except ImportError:
        pass

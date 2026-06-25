"""tytonai_utils — reusable helpers for interacting with the tytonai platform.

Features (built incrementally):
1. Web map import   — download all tiles of a web map from a tile size + webmap link.
2. Manifest import  — download tiles listed in a manifest (image + mask .npz).
3. Model fetch      — download a model from its config file.
4. Mask rollup      — remap annotation mask categories.
"""

__version__ = "0.1.0"

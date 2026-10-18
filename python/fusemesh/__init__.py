"""FUSE: Fast Union Surface Extraction.

M0 exposes build diagnostics only; reconstruction arrives in M1.
"""
from ._native import build_info, smoke_test

__version__ = "0.0.1.dev0"
__all__ = ["build_info", "smoke_test"]

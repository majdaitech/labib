"""LABIB - Lightweight AI Agent SDK

This package re-exports the public API from the original `majdk` package to
provide a seamless rename while maintaining compatibility.
"""
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from majdk import *  # noqa: F401,F403
    from majdk import __all__ as _MAJDK_ALL, __version__ as __version__

__all__ = _MAJDK_ALL

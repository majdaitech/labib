"""LABIB CLI proxy.

This module forwards to the original majdk CLI while the codebase is in
transition. It allows `labib` console script entry point to work.
"""
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from majdk.cli import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

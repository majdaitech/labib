"""LABIB Streamlit UI proxy.

This forwards to the original majdk Streamlit UI to support the `labib-ui`
console script during the rename.
"""
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from majdk.ui_app import run_app as _run_app


def run_app() -> None:
    _run_app()


if __name__ == "__main__":
    run_app()

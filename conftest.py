"""
Root conftest.py — project-wide pytest configuration.

Custom marks
------------
slow : tests that take > ~30 seconds (e.g. full MC pilots).
       Skip with:  pytest -m "not slow"
       Run only:   pytest -m slow
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (run with -m slow; skipped by -m 'not slow')",
    )

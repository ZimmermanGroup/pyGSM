"""
Unit and regression test for the pyGSM package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pyGSM


def test_pyGSM_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pyGSM" in sys.modules

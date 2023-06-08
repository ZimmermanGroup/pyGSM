"""
Unit and regression test for the pygsm package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pygsm


def test_pygsm_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pygsm" in sys.modules

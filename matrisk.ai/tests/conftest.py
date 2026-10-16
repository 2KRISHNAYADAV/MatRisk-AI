"""
tests/conftest.py
------------------
Shared fixtures for the MatRisk AI dashboard tests.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def processed_dir(project_root):
    return project_root / "data" / "processed"


@pytest.fixture(scope="session")
def models_dir(project_root):
    return project_root / "data" / "processed" / "models"


@pytest.fixture(scope="session")
def tables_dir(project_root):
    return project_root / "outputs" / "tables"


@pytest.fixture(scope="session")
def figures_dir(project_root):
    return project_root / "outputs" / "figures"

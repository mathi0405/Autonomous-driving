"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``helpers`` importable regardless of pytest's import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from helpers import make_cfg  # noqa: E402


@pytest.fixture
def cfg():
    """A fast, state-observation config."""
    return make_cfg("state")


@pytest.fixture
def image_cfg():
    """A fast, single-frame image-observation config."""
    return make_cfg("image", frame_stack=1)

"""The shared test doubles (``painted_mock``, ``fake_model``) live here, beside the
folders that mirror the package."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _a_test_key(monkeypatch):
    """The default channel (OpenAI) wants a key; the tests only ever talk to stand-ins."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

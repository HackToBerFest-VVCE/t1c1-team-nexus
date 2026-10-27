"""Shared pytest setup: project root on sys.path + clean store/clock per test."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import incident_store as store


@pytest.fixture(autouse=True)
def _reset_store():
    store.reset()
    store.set_clock(time.time)
    store.set_ai_enabled(True)
    store.set_auto_accept(False)
    yield
    store.reset()
    store.set_clock(time.time)

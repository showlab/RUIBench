"""The interfaces read the same whatever the machine's locale: an interface's
title has a middle dot in it, and after Isaac starts the process's default
text encoding can be ASCII (a worker's second interface once failed to load)."""

from pathlib import Path


def test_the_profiles_load_where_the_default_text_encoding_is_ascii(monkeypatch):
    from ruikit.interfaces import catalog

    real = Path.read_text

    def strict(self, encoding=None, errors=None):
        return real(self, encoding=encoding or "ascii", errors=errors)

    monkeypatch.setattr(Path, "read_text", strict)
    catalog._catalog.cache_clear()                    # read it again, under the strict reading
    try:
        for name in ("rui.step/v1", "rui.pose/v1", "rui.point/v1", "rui.code/v1"):
            assert "·" in catalog.load_profile(name).title
    finally:
        catalog._catalog.cache_clear()

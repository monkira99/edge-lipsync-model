from __future__ import annotations


def test_package_imports() -> None:
    import edge_lipsync

    assert edge_lipsync.__version__ == "0.1.0"

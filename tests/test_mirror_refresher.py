import json
from pathlib import Path

from fetcher.tools.mirror_refresher import refresh_manifest


def test_refresh_manifest_writes_files(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"url": "https://example.com/doc", "path": "docs/output.txt"}]), encoding="utf-8")

    def fake_fetch(url: str) -> bytes:
        return f"mirror:{url}".encode("utf-8")

    out_dir = tmp_path / "local"
    stats = refresh_manifest(manifest, out_dir, fetch=fake_fetch)

    target = out_dir / "docs/output.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "mirror:https://example.com/doc"
    assert stats[0]["sha256"]

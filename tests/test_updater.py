import io
import os
import sys
import zipfile

import pytest

from cta import updater


def test_versions():
    assert updater.parse_version("v1.2.10") == (1, 2, 10)
    assert updater.is_newer("1.0.1", "1.0.0")
    assert updater.is_newer("1.10.0", "1.9.9")
    assert not updater.is_newer("1.0.0", "1.0.0")
    assert not updater.is_newer("0.9", "1.0.0")


RELEASE = {
    "tag_name": "v1.2.0", "body": "notes", "html_url": "https://x/rel", "draft": False, "prerelease": False,
    "assets": [
        {"name": "CTA-1.2.0-windows-x64.zip", "browser_download_url": "https://x/w.zip", "size": 10},
        {"name": "CTA-1.2.0-macos-arm64.zip", "browser_download_url": "https://x/m.zip", "size": 20},
        {"name": "SHA256SUMS.txt", "browser_download_url": "https://x/sums"},
    ],
}


def test_release_asset_selection():
    r = updater.release_from_json(RELEASE, "windows-x64")
    assert r.version == "1.2.0" and r.asset_url == "https://x/w.zip" and r.installable
    r = updater.release_from_json(RELEASE, "macos-arm64")
    assert r.asset_name == "CTA-1.2.0-macos-arm64.zip"
    r = updater.release_from_json(RELEASE, None)
    assert not r.installable


def test_parse_sums():
    h = "a" * 64
    sums = updater.parse_sums(f"{h}  CTA-1.2.0-windows-x64.zip\n{'b' * 64} *other.zip\ngarbage\n")
    assert sums == {"CTA-1.2.0-windows-x64.zip": h, "other.zip": "b" * 64}


def test_not_frozen_cannot_self_update():
    ok, why = updater.can_self_update()
    assert not ok and "source" in why


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS extraction uses ditto")
def test_extract_rejects_path_traversal(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    p = tmp_path / "u.zip"
    p.write_bytes(buf.getvalue())
    with pytest.raises(updater.UpdateError):
        updater.extract(str(p), str(tmp_path / "out"))
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS extraction uses ditto")
def test_extract_layout(tmp_path):
    p = tmp_path / "u.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("CTA/marker.txt", "new")
    new = updater.extract(str(p), str(tmp_path / "out"))
    assert os.path.basename(new) == "CTA"
    assert open(os.path.join(new, "marker.txt")).read() == "new"


def test_helper_scripts_have_rollback():
    ps = updater.windows_script(123, r"C:\Apps\CTA", r"C:\Apps\.cta-update-1\payload\CTA",
                                r"C:\Apps\.cta-update-1", r"C:\log.txt", True)
    assert "Wait-Process -Id 123" in ps and "UPDATE OK" in ps and "Move-Item -LiteralPath $old" in ps
    sh = updater.posix_script(123, "/a/CTA.app", "/a/.s/payload/CTA.app", "/a/.s", "/l", False)
    assert 'mv "$TARGET.old" "$TARGET"' in sh and "UPDATE OK" in sh

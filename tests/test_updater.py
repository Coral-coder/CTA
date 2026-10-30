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


def test_mac_update_target_escapes_translocation(tmp_path, monkeypatch):
    """An app opened from Downloads runs from a read-only translocated copy:
    the update installs to an Applications folder instead."""
    apps = tmp_path / "Applications"
    apps.mkdir()
    monkeypatch.setattr(updater.sys, "platform", "darwin")
    monkeypatch.setattr(updater, "install_root",
                        lambda: "/private/var/folders/x/AppTranslocation/ABC/d/CTA.app")
    real_makedirs = os.makedirs
    monkeypatch.setattr(updater.os, "makedirs",
                        lambda p, exist_ok=False: real_makedirs(str(apps) if p == "/Applications" else p,
                                                                exist_ok=exist_ok))
    dest = os.path.join("/Applications", "CTA.app")          # the separator differs on Windows runners
    monkeypatch.setattr(updater, "_writable", lambda p: p == dest)
    assert updater.update_target() == dest
    # a normal, writable install updates in place
    monkeypatch.setattr(updater, "install_root", lambda: dest)
    assert updater.update_target() == dest


@pytest.mark.skipif(sys.platform == "win32", reason="bash helper")
def test_posix_helper_installs_to_a_new_place(tmp_path):
    import subprocess

    new = tmp_path / "stage" / "payload" / "CTA"
    new.mkdir(parents=True)
    (new / "marker").write_text("v2")
    target = tmp_path / "Apps" / "CTA"                 # not there yet (first update from a read-only copy)
    log = tmp_path / "update.log"
    sh = updater.posix_script(999999, str(target), str(new), str(tmp_path / "stage"), str(log), False)
    script = tmp_path / "u.sh"
    script.write_text(sh)
    subprocess.run(["bash", str(script)], check=True, timeout=60)
    assert (target / "marker").read_text() == "v2" and "UPDATE OK" in log.read_text()
    # and the next update swaps in place
    new.mkdir(parents=True)
    (new / "marker").write_text("v3")
    subprocess.run(["bash", str(script)], check=True, timeout=60)
    assert (target / "marker").read_text() == "v3"

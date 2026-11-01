"""Self-update from GitHub Releases.

Flow: check the latest release -> download the zip for this platform ->
verify its SHA-256 against the release's SHA256SUMS.txt -> unpack next to
the install -> run the new build once (``--version``) to prove it starts ->
hand over to a tiny helper script that waits for Tempest to exit, swaps the
install folders (keeping the old one until the swap succeeded, rolling back
otherwise) and relaunches.

User data (library database, settings, models) lives outside the install
folder and is never touched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass

from . import APP_NAME, LEGACY_NAME, __version__, net, paths

log = logging.getLogger(__name__)

REPO = "Coral-coder/CTA"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
USER_AGENT = f"Tempest/{__version__}"


class UpdateError(Exception):
    pass


@dataclass
class Release:
    version: str
    tag: str
    notes: str
    url: str                  # release web page
    asset_name: str | None
    asset_url: str | None
    asset_size: int
    sums_url: str | None

    @property
    def installable(self) -> bool:
        return bool(self.asset_url and self.sums_url)


# --------------------------------------------------------------------------
# version / platform helpers
# --------------------------------------------------------------------------

def parse_version(v: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", v.split("-")[0].split("+")[0])
    return tuple(int(n) for n in nums[:4]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def platform_target() -> str | None:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x64"
    if sys.platform == "darwin" and machine == "arm64":
        return "macos-arm64"
    return None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_root() -> str | None:
    """Folder (Windows/Linux) or .app bundle (macOS) that holds this build."""
    if not is_frozen():
        return None
    exe = os.path.abspath(sys.executable)
    if sys.platform == "darwin":
        m = re.match(r"(.*?\.app)/Contents/MacOS/", exe)
        return m.group(1) if m else None
    return os.path.dirname(exe)


def _writable(path: str) -> bool:
    parent = os.path.dirname(path)
    return os.access(parent, os.W_OK) and (not os.path.exists(path) or os.access(path, os.W_OK))


def update_target() -> str | None:
    """Where the new version goes. Normally where Tempest runs from. On a Mac,
    an app opened straight from Downloads runs from a hidden read-only copy
    ("App Translocation") or a place the user can't write to: then the update
    goes to /Applications (or ~/Applications) instead - and from there on
    Tempest updates itself in place. A version started by a live update runs
    from a copy: the install it replaced is still the target."""
    if os.environ.get("TEMPEST_INSTALL_TARGET"):
        return os.environ["TEMPEST_INSTALL_TARGET"]
    root = install_root()
    if root is None:
        return None
    if sys.platform != "darwin":
        return root
    if "/AppTranslocation/" not in root and not root.startswith("/Volumes/") and _writable(root):
        return root
    for folder in ("/Applications", os.path.expanduser("~/Applications")):
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            continue
        dest = os.path.join(folder, f"{APP_NAME}.app")
        if _writable(dest):
            return dest
    return None


def can_self_update(need_release_build: bool = True) -> tuple[bool, str]:
    root = install_root()
    if root is None:
        return False, "Running from source - update with git/pip instead."
    if need_release_build and platform_target() is None:
        return False, "No prebuilt update for this platform."
    target = update_target()
    if target is None or not _writable(target):
        parent = os.path.dirname(target or root)
        return False, f"No write permission for {parent}. Move Tempest to a folder you own, or update manually."
    return True, ""


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/vnd.github+json"})
    with net.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def release_from_json(d: dict, target: str | None = "auto") -> Release:
    """``target`` is a platform name, None (no build wanted) or "auto"."""
    if target == "auto":
        target = platform_target()
    tag = d.get("tag_name") or ""
    asset_name = asset_url = sums_url = None
    size = 0
    for a in d.get("assets", []):
        name = a.get("name", "")
        if name == "SHA256SUMS.txt":
            sums_url = a.get("browser_download_url")
        elif target and name.endswith(f"-{target}.zip") and not name.startswith(("legacy-", "upgrade-from-")):
            # the Tempest zip wins; the upgrade-from-CTA one (for installs still called CTA, whose
            # updater takes the last matching asset - GitHub sorts names case-insensitively) is ignored
            if asset_name is None or name.startswith(f"{APP_NAME}-"):
                asset_name, asset_url, size = name, a.get("browser_download_url"), int(a.get("size") or 0)
    return Release(version=tag.lstrip("vV"), tag=tag, notes=d.get("body") or "",
                   url=d.get("html_url") or RELEASES_PAGE, asset_name=asset_name, asset_url=asset_url,
                   asset_size=size, sums_url=sums_url)


def check_latest(timeout: float = 15) -> Release:
    try:
        d = _get_json(API_LATEST, timeout)
    except Exception as e:  # noqa: BLE001
        raise UpdateError(f"could not reach GitHub: {e}") from e
    if d.get("draft") or d.get("prerelease"):
        raise UpdateError("latest release is not a stable release")
    return release_from_json(d)


# --------------------------------------------------------------------------
# download + verify + stage
# --------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def _download(url: str, dest: str, progress=None, cancelled=None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with net.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancelled and cancelled():
                raise UpdateError("cancelled")
            chunk = r.read(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


def staging_dir(version: str) -> str:
    root = install_root()
    base = os.path.dirname(root) if root else tempfile.gettempdir()
    # same volume as the install so the final swap is an atomic rename
    return os.path.join(base, f".cta-update-{version}")


def download_release(rel: Release, progress=None, cancelled=None) -> str:
    """Download and checksum-verify the release zip; returns its path."""
    if not rel.installable:
        raise UpdateError("this release has no build for your platform")
    stage = staging_dir(rel.version)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    sums_req = urllib.request.Request(rel.sums_url, headers={"User-Agent": USER_AGENT})
    try:
        with net.urlopen(sums_req, timeout=30) as r:
            sums = parse_sums(r.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        raise UpdateError(f"could not download checksums: {e}") from e
    expected = sums.get(rel.asset_name)
    if not expected:
        raise UpdateError("release checksum list does not cover this download")
    zpath = os.path.join(stage, rel.asset_name)
    last = None
    for attempt in range(3):
        try:
            _download(rel.asset_url, zpath, progress, cancelled)
            last = None
            break
        except UpdateError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    if last is not None:
        raise UpdateError(f"download failed: {last}")
    if sha256_file(zpath) != expected:
        os.remove(zpath)
        raise UpdateError("downloaded file failed checksum verification")
    return zpath


def _payload_names() -> tuple[str, ...]:
    """Top-level folder of a release zip: the current name, or the old one."""
    if sys.platform == "darwin":
        return (f"{APP_NAME}.app", f"{LEGACY_NAME}.app")
    return (APP_NAME, LEGACY_NAME)


def extract(zpath: str, dest: str) -> str:
    """Unpack the release zip into ``dest``; returns the new install path."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    if sys.platform == "darwin":
        # ditto keeps symlinks, permissions and code signatures intact
        r = subprocess.run(["ditto", "-x", "-k", zpath, dest], capture_output=True, text=True)
        if r.returncode != 0:
            raise UpdateError(f"unpack failed: {r.stderr.strip()}")
    else:
        with zipfile.ZipFile(zpath) as z:
            for m in z.infolist():
                target = os.path.realpath(os.path.join(dest, m.filename))
                if not target.startswith(os.path.realpath(dest) + os.sep):
                    raise UpdateError("unsafe path in update archive")
            z.extractall(dest)
            if sys.platform != "win32":
                for m in z.infolist():
                    mode = (m.external_attr >> 16) & 0o777
                    if mode:
                        os.chmod(os.path.join(dest, m.filename), mode)
    for name in _payload_names():
        new = os.path.join(dest, name)
        if os.path.isdir(new):
            return new
    raise UpdateError("update archive has an unexpected layout")


def _cli_path(app_root: str) -> str:
    """The command-line entry of an install (new name, or the old one in a legacy install)."""
    if sys.platform == "darwin":
        names = [os.path.join(app_root, "Contents", "MacOS", n) for n in (APP_NAME, LEGACY_NAME)]
    elif sys.platform == "win32":
        names = [os.path.join(app_root, n) for n in ("tempest-cli.exe", "cta-cli.exe")]
    else:
        names = [os.path.join(app_root, n) for n in ("tempest-cli", "cta-cli")]
    return next((n for n in names if os.path.exists(n)), names[0])


def _mac_signature(app: str) -> dict | None:
    """{'authority': [...], 'team': 'XXXX'} of a signed Mac app bundle, or None."""
    if sys.platform != "darwin":
        return None
    try:
        r = subprocess.run(["codesign", "-dvv", app], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    out = r.stdout + r.stderr
    auth = re.findall(r"^Authority=(.+)$", out, re.M)
    team = re.search(r"^TeamIdentifier=(.+)$", out, re.M)
    if not auth or not team or team.group(1).strip() in ("", "not set"):
        return None
    return {"authority": auth, "team": team.group(1).strip()}


def check_signature(new_root: str) -> str:
    """macOS: when the running app carries a Developer ID signature, the new
    one must too - valid, and from the same team - or the update is refused.
    (An unsigned / ad-hoc install accepts any build, as before.)"""
    mine = _mac_signature(install_root() or "")
    if mine is None or not any(a.startswith("Developer ID") for a in mine["authority"]):
        return "unsigned"
    theirs = _mac_signature(new_root)
    if theirs is None or theirs["team"] != mine["team"]:
        raise UpdateError("the downloaded app is not signed by the same developer - update refused")
    r = subprocess.run(["codesign", "--verify", "--deep", "--strict", new_root], capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise UpdateError(f"the downloaded app's signature does not verify: {(r.stderr or r.stdout)[-300:]}")
    return f"signed by team {mine['team']}"


def verify_build(new_root: str, expect_version: str | None) -> str:
    """Run the new build once; it must start and report its version."""
    exe = _cli_path(new_root)
    if not os.path.isfile(exe):
        raise UpdateError("new build is missing its executable")
    env = dict(os.environ)
    env["CTA_DATA_DIR"] = tempfile.mkdtemp(prefix="cta-verify-")
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=120, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:  # noqa: BLE001
        raise UpdateError(f"new build failed to start: {e}") from e
    out = (r.stdout + r.stderr).strip()
    m = re.search(r"Tempest (\S+)", out)
    if r.returncode != 0 or not m:
        raise UpdateError(f"new build failed to start: {out[-300:]}")
    if expect_version and parse_version(m.group(1)) != parse_version(expect_version):
        raise UpdateError(f"new build reports version {m.group(1)}, expected {expect_version}")
    return m.group(1)


# --------------------------------------------------------------------------
# swap (performed by a helper after Tempest exits)
# --------------------------------------------------------------------------

def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def windows_script(pid: int, target: str, new: str, stage: str, log_file: str, relaunch: bool) -> str:
    gui = os.path.join(target, f"{APP_NAME}.exe")
    return f"""$ErrorActionPreference = 'Stop'
try {{ Start-Transcript -LiteralPath {_ps_quote(log_file + ".transcript")} -Append | Out-Null }} catch {{ }}
$log = {_ps_quote(log_file)}
function L($m) {{ Add-Content -LiteralPath $log -Value ("{{0}} {{1}}" -f (Get-Date -Format s), $m) }}
$target = {_ps_quote(target)}
$new = {_ps_quote(new)}
$stage = {_ps_quote(stage)}
$old = $target + '.old'
$relaunch = ${'true' if relaunch else 'false'}
try {{
    L "waiting for Tempest (pid {pid}) to exit"
    try {{ Wait-Process -Id {pid} -Timeout 180 -ErrorAction Stop }} catch {{ }}
    Start-Sleep -Milliseconds 800
    if (Test-Path -LiteralPath $old) {{ Remove-Item -LiteralPath $old -Recurse -Force }}
    $moved = $false
    for ($i = 0; $i -lt 40; $i++) {{
        try {{ Move-Item -LiteralPath $target -Destination $old; $moved = $true; break }}
        catch {{ Start-Sleep -Milliseconds 750 }}
    }}
    if (-not $moved) {{ throw "the Tempest folder is still in use" }}
    try {{ Move-Item -LiteralPath $new -Destination $target }}
    catch {{ Move-Item -LiteralPath $old -Destination $target; throw }}
    L "UPDATE OK"
    Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
}} catch {{
    L ("UPDATE FAILED: " + $_)
}}
if ($relaunch) {{ Start-Process -FilePath {_ps_quote(gui)} }}
"""


def posix_script(pid: int, target: str, new: str, stage: str, log_file: str, relaunch: bool) -> str:
    launch = (f"open {_sh_quote(target)}" if sys.platform == "darwin"
              else f"nohup {_sh_quote(os.path.join(target, APP_NAME))} >/dev/null 2>&1 &")
    # macOS: a downloaded app is "quarantined" and run from a hidden read-only copy;
    # clearing the flag lets the installed app run (and update) from where it is
    unq = 'xattr -dr com.apple.quarantine "$TARGET" 2>/dev/null' if sys.platform == "darwin" else ":"
    return f"""#!/bin/bash
LOG={_sh_quote(log_file)}
TARGET={_sh_quote(target)}
NEW={_sh_quote(new)}
STAGE={_sh_quote(stage)}
log() {{ echo "$(date '+%Y-%m-%dT%H:%M:%S') $1" >> "$LOG"; }}
log "waiting for Tempest (pid {pid}) to exit"
for i in $(seq 1 360); do kill -0 {pid} 2>/dev/null || break; sleep 0.5; done
sleep 0.5
rm -rf "$TARGET.old"
if [ ! -e "$TARGET" ]; then
    # first update from a read-only / downloaded copy: install to the new place
    mkdir -p "$(dirname "$TARGET")"
    if mv "$NEW" "$TARGET"; then
        log "UPDATE OK (installed to $TARGET)"
        rm -rf "$STAGE"
    else
        log "UPDATE FAILED: could not move the new version into place"
    fi
elif mv "$TARGET" "$TARGET.old"; then
    if mv "$NEW" "$TARGET"; then
        log "UPDATE OK"
        rm -rf "$TARGET.old" "$STAGE"
    else
        mv "$TARGET.old" "$TARGET"
        log "UPDATE FAILED: could not move the new version into place"
    fi
else
    log "UPDATE FAILED: could not move the old version aside"
fi
{unq}
{launch if relaunch else ":"}
"""


def update_log_path() -> str:
    return os.path.join(paths.data_dir(), "update.log")


def launch_swap(new_root: str, relaunch: bool = True, pid: int | None = None) -> None:
    """Start the helper; the caller must then exit the application."""
    target = update_target()
    if target is None:
        raise UpdateError("not running from an installed build")
    pid = pid or os.getpid()
    stage = os.path.dirname(os.path.dirname(new_root))     # <stage>/payload/<Tempest or Tempest.app>
    log_file = update_log_path()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} staged {new_root} -> {target}\n")
    tmp = tempfile.mkdtemp(prefix="cta-updater-")
    if sys.platform == "win32":
        script = os.path.join(tmp, "update.ps1")
        with open(script, "w", encoding="utf-8-sig") as f:
            f.write(windows_script(pid, target, new_root, stage, log_file, relaunch))
        args = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script]
        base = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        breakaway = 0x01000000   # CREATE_BREAKAWAY_FROM_JOB: survive the parent's job object closing
        try:
            subprocess.Popen(args, cwd=tmp, creationflags=base | breakaway, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            # the job does not allow breakaway; a plain hidden child still works
            subprocess.Popen(args, cwd=tmp, creationflags=base, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        script = os.path.join(tmp, "update.sh")
        with open(script, "w", encoding="utf-8") as f:
            f.write(posix_script(pid, target, new_root, stage, log_file, relaunch))
        os.chmod(script, 0o755)
        subprocess.Popen(["/bin/bash", script], cwd=tmp, start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_update(rel: Release, progress=None, cancelled=None) -> str:
    """Download, verify, unpack and test-run; returns the staged install."""
    ok, why = can_self_update()
    if not ok:
        raise UpdateError(why)
    zpath = download_release(rel, progress, cancelled)
    new_root = extract(zpath, os.path.join(os.path.dirname(zpath), "payload"))
    os.remove(zpath)
    log.info("update signature: %s", check_signature(new_root))
    verify_build(new_root, rel.version)
    return new_root


def prepare_from_zip(zpath: str, expect_version: str | None = None) -> str:
    """Stage an update from a local zip (manual installs and CI tests)."""
    ok, why = can_self_update(need_release_build=False)
    if not ok:
        raise UpdateError(why)
    stage = staging_dir("local")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    new_root = extract(zpath, os.path.join(stage, "payload"))
    log.info("update signature: %s", check_signature(new_root))
    verify_build(new_root, expect_version)
    return new_root


def cleanup_after_update() -> str | None:
    """Called at start-up: report the result of the last update and remove
    leftovers. Returns 'ok', 'failed: ...' or None."""
    root = install_root()
    if root:
        for leftover in (root + ".old",):
            if os.path.isdir(leftover):
                shutil.rmtree(leftover, ignore_errors=True)
    lp = update_log_path()
    if not os.path.exists(lp):
        return None
    try:
        lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    result = None
    for line in reversed(lines):
        if "UPDATE OK" in line:
            result = "ok"
            break
        if "UPDATE FAILED" in line:
            result = "failed: " + line.split("UPDATE FAILED:", 1)[-1].strip()
            break
    try:
        os.replace(lp, lp + ".last")
    except OSError:
        pass
    return result

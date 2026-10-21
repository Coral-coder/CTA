"""Download and verify the AI model for bundling (used by CI)."""

import hashlib
import os
import re
import sys
import time
import urllib.request

# read the constants without importing cta (dependencies may not be installed yet)
_src = open(os.path.join(os.path.dirname(__file__), "..", "src", "cta", "model.py"), encoding="utf-8").read()
MODEL_FILE = re.search(r'^MODEL_FILE = "([^"]+)"', _src, re.M).group(1)
MODEL_SHA256 = re.search(r'^MODEL_SHA256 = "([0-9a-f]{64})"', _src, re.M).group(1)
MODEL_URL = re.search(r'^MODEL_URL = "([^"]+)" \+ MODEL_FILE', _src, re.M).group(1) + MODEL_FILE


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, MODEL_FILE)
    if os.path.exists(dest) and sha256(dest) == MODEL_SHA256:
        print("model already present")
        return 0
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "CTA-CI"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            if sha256(dest + ".part") != MODEL_SHA256:
                raise RuntimeError("checksum mismatch")
            os.replace(dest + ".part", dest)
            print(f"downloaded {dest}")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build_models"))

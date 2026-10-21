"""Download and verify the AI model for bundling (used by CI)."""

import hashlib
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cta.model import MODEL_FILE, MODEL_SHA256, MODEL_URL  # noqa: E402


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

"""Turn the painted icon (packaging/icon/tempest.png, from make_storm.py) into
icon.ico (Windows), icon.icns (macOS) and the in-app and phone icons:

    python packaging/make_icons.py
"""

import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def write_ico(path: str, images: dict[int, bytes]):
    """ICO with PNG-compressed entries (Windows Vista+)."""
    sizes = sorted(images)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, blobs = b"", b""
    for s in sizes:
        data = images[s]
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset + len(blobs))
        blobs += data
    with open(path, "wb") as f:
        f.write(header + entries + blobs)


def write_icns(path: str, images: dict[int, bytes]):
    types = {16: b"icp4", 32: b"icp5", 64: b"icp6", 128: b"ic07", 256: b"ic08", 512: b"ic09", 1024: b"ic10"}
    body = b""
    for s, t in types.items():
        if s in images:
            body += t + struct.pack(">I", 8 + len(images[s])) + images[s]
    with open(path, "wb") as f:
        f.write(b"icns" + struct.pack(">I", 8 + len(body)) + body)


def scaled_png(src: QImage, size: int, opaque: bool = False) -> bytes:
    img = src.scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    if opaque:
        img = img.convertToFormat(QImage.Format_RGB32)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def main() -> int:
    """The icon is painted (icon/make_storm.py writes tempest.png and
    tempest-square.png); this turns it into every format the app needs."""
    _app = QGuiApplication(sys.argv)
    tile = QImage(os.path.join(HERE, "icon", "tempest.png"))
    square = QImage(os.path.join(HERE, "icon", "tempest-square.png"))
    if tile.isNull() or square.isNull():
        print("run packaging/icon/make_storm.py first")
        return 1
    imgs = {s: scaled_png(tile, s) for s in (16, 24, 32, 48, 64, 128, 256, 512, 1024)}
    write_ico(os.path.join(HERE, "icon.ico"), {s: imgs[s] for s in (16, 24, 32, 48, 64, 128, 256)})
    write_icns(os.path.join(HERE, "icon.icns"), imgs)
    with open(os.path.join(ROOT, "src", "cta", "icon.png"), "wb") as f:
        f.write(imgs[256])
    for size in (180, 512):              # apple-touch-icon / web app manifest: full bleed, the phone rounds it
        with open(os.path.join(ROOT, "src", "cta", f"app-icon-{size}.png"), "wb") as f:
            f.write(scaled_png(square, size, opaque=True))
    print("icons written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

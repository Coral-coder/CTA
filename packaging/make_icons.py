"""Render packaging/icon/tempest.svg into icon.ico (Windows), icon.icns (macOS)
and the in-app window icon. Run after editing the SVG:

    python packaging/make_icons.py
"""

import os
import re
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def png_bytes(renderer: QSvgRenderer, size: int, opaque: bool = False) -> bytes:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.black if opaque else Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer.render(p)
    p.end()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def app_svg(src: str) -> str:
    """Home-screen version (iPhone/Android): full-bleed square (the phone
    rounds the corners) and the artwork a little larger."""
    s = re.sub(r'\s*<rect x="32" y="32"[^>]*/>', "", src)
    i = s.index("<!-- tile -->")
    j = s.rindex("</svg>")
    return (s[:i] + '<rect x="0" y="0" width="1024" height="1024" fill="url(#bg)"/>\n'
            '<g transform="translate(512 512) scale(1.08) translate(-512 -512)">' + s[i:j] + "</g>\n" + s[j:])


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


def main() -> int:
    _app = QGuiApplication(sys.argv)
    r = QSvgRenderer(os.path.join(HERE, "icon", "tempest.svg"))
    if not r.isValid():
        print("invalid SVG")
        return 1
    imgs = {s: png_bytes(r, s) for s in (16, 24, 32, 48, 64, 128, 256, 512, 1024)}
    write_ico(os.path.join(HERE, "icon.ico"), {s: imgs[s] for s in (16, 24, 32, 48, 64, 128, 256)})
    write_icns(os.path.join(HERE, "icon.icns"), imgs)
    with open(os.path.join(ROOT, "src", "cta", "icon.png"), "wb") as f:
        f.write(imgs[256])
    with open(os.path.join(HERE, "icon", "tempest.svg"), encoding="utf-8") as f:
        ra = QSvgRenderer(QByteArray(app_svg(f.read()).encode()))
    for size in (180, 512):              # apple-touch-icon / web app manifest
        with open(os.path.join(ROOT, "src", "cta", f"app-icon-{size}.png"), "wb") as f:
            f.write(png_bytes(ra, size, opaque=True))
    print("icons written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

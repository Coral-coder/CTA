"""Paint the Tempest icon: a storm seen from above, spiralling round a vinyl
record that sits in its eye. Natural colours (cloud white, storm grey, slate),
lit from the top left so the cloud bands read in depth. Writes

    packaging/icon/tempest.png          1024 px, rounded tile (desktop icons)
    packaging/icon/tempest-square.png   1024 px, full bleed (phone home screens)

Run once after changing it (needs numpy, scipy, Pillow), then make_icons.py.
"""

import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1024                       # output size
SS = 2                         # supersampling for clean edges
M = N * SS
rng = np.random.default_rng(7)


def fbm(size: int, octaves=(4, 8, 16, 32, 64, 128), gain: float = 0.55) -> np.ndarray:
    """Smooth fractal noise in [0, 1]."""
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for cells in octaves:
        g = rng.random((cells + 3, cells + 3))
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        out += amp * np.asarray(img, dtype=np.float64) / 255.0
        total += amp
        amp *= gain
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def smooth(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    t = t[..., None]
    return a * (1 - t) + b * t


def paint(full_bleed: bool) -> Image.Image:
    yy, xx = np.mgrid[0:M, 0:M].astype(np.float64)
    c = (M - 1) / 2.0
    x, y = (xx - c) / (M / 2), (yy - c) / (M / 2)          # -1..1
    r = np.hypot(x, y)
    th = np.arctan2(y, x)

    # ---- a hurricane as a satellite sees it: arms of cloud spiralling into the eye ----
    eye_r = 0.33
    swirl = th + 3.4 * np.exp(-r * 2.2)                     # the wind twists the texture round the eye
    sx = (np.cos(swirl) * r * 0.5 + 0.5) * (M - 1)
    sy = (np.sin(swirl) * r * 0.5 + 0.5) * (M - 1)
    n1 = map_coordinates(fbm(M, octaves=(6, 12, 24, 48, 96), gain=0.55), [sy, sx], order=1, mode="mirror")
    n2 = map_coordinates(fbm(M, octaves=(24, 48, 96, 192), gain=0.55), [sy, sx], order=1, mode="mirror")
    lr_ = np.log(r + 0.04)
    # two main arms + thinner feeder bands, each a logarithmic spiral
    main = 0.5 + 0.5 * np.cos(2 * th - 5.6 * lr_ + 1.8 * (n1 - 0.5))
    feed = 0.5 + 0.5 * np.cos(5 * th - 9.5 * lr_ + 2.5 * (n1 - 0.5))
    arms = smooth(0.50, 0.92, main) * (1.05 - smooth(0.55, 1.05, r)) \
        + 0.55 * smooth(0.62, 0.95, feed) * smooth(0.45, 0.70, r) * (1 - smooth(0.85, 1.15, r))
    # the dense cloud ring round the eye
    cdo = smooth(eye_r - 0.01, eye_r + 0.07, r) * (1 - smooth(0.50, 0.68, r))
    density = np.clip(np.maximum(cdo * (0.85 + 0.25 * n1), arms * (0.55 + 0.6 * n1)), 0, 1)
    density = density * (0.82 + 0.30 * n2)                  # texture in the cloud tops
    density = gaussian_filter(density, 1.4 * SS)
    density = density * smooth(eye_r - 0.02, eye_r + 0.03, r)

    # ---- light: soft sun from the top left on the cloud tops ----
    h = density * 22.0
    gy, gx = np.gradient(gaussian_filter(h, 2 * SS))
    nz = np.ones_like(h)
    norm = np.sqrt(gx * gx + gy * gy + nz * nz)
    lx, ly, lz = -0.5, -0.6, 0.62
    lam = np.clip((-gx * lx - gy * ly + nz * lz) / norm, 0, 1)
    # the clouds' shadow on the sea, offset away from the sun
    shadow = gaussian_filter(np.roll(np.roll(density, 14 * SS, 0), 12 * SS, 1), 6 * SS)

    sea_deep = np.array([0.055, 0.105, 0.140])
    sea_lit = np.array([0.090, 0.165, 0.205])
    sea = mix(sea_deep, sea_lit, smooth(0.0, 1.0, 1 - r * 0.6) * (0.6 + 0.4 * n2))
    sea = sea * (1 - 0.45 * shadow)[..., None]
    cloud_dark = np.array([0.52, 0.56, 0.58])
    cloud_lit = np.array([0.95, 0.955, 0.95])
    cloud = mix(cloud_dark, cloud_lit, smooth(0.25, 0.95, lam * (0.65 + 0.45 * density)))
    col = mix(sea, cloud, smooth(0.05, 0.75, density))
    # the eye: calm, dark water, the eyewall falling away into it
    eye = mix(np.array([0.04, 0.075, 0.10]), sea_deep, smooth(0.0, eye_r, r))
    col = mix(col, eye, 1 - smooth(eye_r - 0.04, eye_r + 0.02, r))
    col = mix(col, col * 0.85, smooth(0.9, 1.3, r))          # darker towards the tile edge
    lam = lam

    # ---- the record in the eye ----
    vr = 0.255
    # its shadow on the eye floor (down-right of the light)
    sh = np.hypot(x - 0.035, y - 0.045)
    col *= (1 - 0.55 * (1 - smooth(vr - 0.02, vr + 0.07, sh)))[..., None]
    on = 1 - smooth(vr - 0.004, vr + 0.004, r)
    grooves = 0.5 + 0.5 * np.sin(r * 430)
    # anisotropic sheen of light across the grooves (two opposite lobes, like real vinyl)
    sheen = np.clip(np.cos(2 * (th - np.arctan2(ly, lx))), 0, 1) ** 6
    vinyl = np.array([0.035, 0.035, 0.038]) + (0.05 * grooves)[..., None] * 0.35
    vinyl = vinyl + (sheen * (0.20 + 0.08 * grooves))[..., None]
    rim = np.exp(-((r - vr) / 0.006) ** 2)
    vinyl = vinyl + (rim * 0.12)[..., None]
    lr = 0.088
    label = np.array([0.86, 0.85, 0.82])          # plain off-white paper
    label_on = 1 - smooth(lr - 0.003, lr + 0.003, r)
    vinyl = mix(vinyl, label * (0.88 + 0.12 * lam.mean()), label_on)
    vinyl = mix(vinyl, np.array([0.45, 0.46, 0.47]), np.exp(-((r - lr * 0.72) / 0.004) ** 2) * label_on)
    hole = 1 - smooth(0.010, 0.014, r)
    vinyl = mix(vinyl, np.array([0.05, 0.05, 0.05]), hole)
    col = mix(col, vinyl, on)

    # ---- tile ----
    rgb = np.clip(col, 0, 1)
    alpha = np.ones((M, M))
    if not full_bleed:
        # rounded square with a margin (as desktop icons are drawn), a faint inner vignette
        m = 32 * SS / (M / 2)
        half, rad = 1 - m, 210 * SS / (M / 2)
        qx, qy = np.abs(x) - (half - rad), np.abs(y) - (half - rad)
        dist = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad
        alpha = 1 - smooth(-1.5 / (M / 2), 1.5 / (M / 2), dist)
        vig = smooth(-0.12, 0.0, dist)
        rgb = rgb * (1 - 0.35 * vig)[..., None]
    img = np.dstack([rgb, alpha])
    out = Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGBA")
    return out.resize((N, N), Image.LANCZOS)


def main():
    paint(False).save(os.path.join(HERE, "tempest.png"))
    paint(True).convert("RGB").save(os.path.join(HERE, "tempest-square.png"))
    print("painted")


if __name__ == "__main__":
    main()

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

    # ---- the storm: noise twisted into a spiral, tighter towards the eye ----
    twist = 4.2 * np.exp(-r * 1.8) + 1.1 * np.log(r + 0.05)
    a = th + twist
    base = fbm(M)
    detail = fbm(M, octaves=(8, 16, 32, 64), gain=0.55)
    sx = (np.cos(a) * r * 0.5 + 0.5) * (M - 1)
    sy = (np.sin(a) * r * 0.5 + 0.5) * (M - 1)
    clouds = map_coordinates(base, [sy, sx], order=1, mode="mirror")
    fine = map_coordinates(detail, [sy * 1.0, sx * 1.0], order=1, mode="mirror")
    # spiral rain bands (four arms) give the hurricane its structure
    arms = 0.5 + 0.5 * np.cos(3 * th + 7.5 * np.log(r + 0.08) + 1.6 * (clouds - 0.5))
    arms = arms ** 1.6
    # billows: ridged noise gives cumulus-like puffs along the bands
    puff = fbm(M, octaves=(12, 24, 48, 96), gain=0.5)
    puff = map_coordinates(puff, [sy, sx], order=1, mode="mirror")
    billow = 1 - np.abs(2 * puff - 1)
    density = 0.40 * clouds + 0.10 * fine + 0.55 * arms + 0.30 * billow * (0.4 + 0.6 * arms)
    # the eye: clear in the middle, a tall wall of cloud round it
    eye_r, wall_r = 0.37, 0.49
    wall = np.exp(-((r - wall_r) / 0.075) ** 2)
    density = density * smooth(eye_r - 0.02, wall_r, r) + 0.55 * wall
    density = gaussian_filter(density, 1.8 * SS)

    # ---- light it: the density is a height field, the sun is up and to the left ----
    h = density * 30.0
    gy, gx = np.gradient(h)
    nz = np.ones_like(h) * 1.0
    norm = np.sqrt(gx * gx + gy * gy + nz * nz)
    lx, ly, lz = -0.55, -0.62, 0.56
    lam = np.clip((-gx * lx - gy * ly + nz * lz) / norm, 0, 1)
    # soft self shadow: low ground in the lee of tall cloud
    shade = gaussian_filter(h, 6 * SS)
    occl = np.clip(1.0 - (np.roll(np.roll(shade, 10 * SS, 0), 10 * SS, 1) - h) * 0.08, 0.55, 1.0)

    deep = np.array([0.105, 0.125, 0.145])        # gaps between the bands: dark sea-slate
    mid = np.array([0.38, 0.42, 0.45])            # storm grey
    top = np.array([0.90, 0.91, 0.90])            # sunlit cloud tops
    warm = np.array([0.95, 0.93, 0.88])           # a touch of late-day warmth on the highlights
    d = smooth(0.18, 0.85, density)
    col = mix(deep, mid, d)
    col = mix(col, top, smooth(0.45, 1.0, d * lam * 1.25))
    col = mix(col, warm, smooth(0.80, 1.0, lam) * d * 0.5)
    col *= (0.42 + 0.62 * lam)[..., None] * occl[..., None]
    # the eye floor: dark, calm water far below
    floor = np.array([0.07, 0.085, 0.10])
    col = mix(col, floor, 1 - smooth(eye_r - 0.03, eye_r + 0.06, r))
    # the outer edge of the storm falls off into dark sky
    sky = np.array([0.09, 0.105, 0.12])
    col = mix(col, sky, smooth(0.86, 1.25, r))

    # ---- the record in the eye ----
    vr = 0.305
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
    lr = 0.105
    label = np.array([0.78, 0.71, 0.60])          # aged paper
    label_on = 1 - smooth(lr - 0.003, lr + 0.003, r)
    vinyl = mix(vinyl, label * (0.88 + 0.12 * lam.mean()), label_on)
    vinyl = mix(vinyl, np.array([0.55, 0.47, 0.38]), np.exp(-((r - lr * 0.72) / 0.004) ** 2) * label_on)
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

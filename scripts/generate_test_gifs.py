#!/usr/bin/env python3
"""Generate the before/after GIFs for the Tests page (tests.html).

Each scenario is rendered twice:
  * before: a hand-built trajectory showing a typical physics failure of raw
            video-generation output (floating, momentum created, self-acceleration)
  * after:  the physically correct trajectory (analytic solution)

Every frame carries the quantity the eval measures (a live chart plus the fitted
value), so the GIF shows what the test checks, not just the motion.

Usage:  python3 scripts/generate_test_gifs.py      (writes assets/tests/*.gif)
Requires Pillow and numpy.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "tests")

W, H = 560, 350          # output size
S = 2                    # supersampling factor
FPS = 24
DUR = 3.0                # seconds per loop
NF = int(FPS * DUR)
PX_PER_M = 70.0          # world scale: 8 m across
FLOOR_Y = 292            # floor line (output px); scenes without tall motion raise it

BG_TOP = (0, 0, 0)
BG_BOT = (10, 12, 17)
TEXT = (245, 247, 250)
MUTED = (150, 156, 168)
FAINT = (92, 98, 110)
ORANGE = (255, 85, 38)
TEAL = (16, 235, 175)
RED = (255, 62, 95)
BLUE = (95, 185, 255)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(MONO_B if bold else MONO, size * S)


F_SMALL, F_SMALL_B, F_MED_B, F_TINY = font(10), font(10, True), font(12, True), font(9)


def wx(x):  # world metres -> supersampled px
    return (40 + x * PX_PER_M) * S


def wy(y):
    return (FLOOR_Y - y * PX_PER_M) * S


# ---------------------------------------------------------------- background
def make_background():
    w, h = W * S, H * S
    ys = np.linspace(0, 1, h)[:, None]
    grad = (np.array(BG_TOP)[None, None, :] * (1 - ys[..., None]) + np.array(BG_BOT)[None, None, :] * ys[..., None])
    img = np.repeat(grad, w, axis=1).astype(np.uint8)
    im = Image.fromarray(img, "RGB")
    d = ImageDraw.Draw(im, "RGBA")
    fy = FLOOR_Y * S
    # floor plane
    d.rectangle([0, fy, w, h], fill=(13, 15, 21, 255))
    # perspective grid on the floor
    vx, vy = w / 2, fy - 260 * S
    for k in range(-14, 15):
        x_bottom = w / 2 + k * 70 * S
        t = (fy - vy) / (h - vy)
        x_top = vx + (x_bottom - vx) * t
        d.line([(x_top, fy), (x_bottom, h)], fill=(255, 255, 255, 14), width=S)
    for k in range(1, 7):
        y = fy + (h - fy) * (k / 6) ** 1.6
        d.line([(0, y), (w, y)], fill=(255, 255, 255, 12 + k * 2), width=S)
    d.line([(0, fy), (w, fy)], fill=(255, 255, 255, 60), width=S)
    return im


_BG_CACHE = {}


def set_floor(y):
    global FLOOR_Y, BG
    FLOOR_Y = y
    if y not in _BG_CACHE:
        _BG_CACHE[y] = make_background()
    BG = _BG_CACHE[y]


# ---------------------------------------------------------------- sprites
def sphere_sprite(r_m, tint, rim):
    r = int(round(r_m * PX_PER_M * S))
    size = 2 * r + 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx, ny = (xx - r - 0.5) / r, (yy - r - 0.5) / r
    d2 = nx * nx + ny * ny
    mask = d2 <= 1.0
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array([-0.55, -0.65, 0.52]); L /= np.linalg.norm(L)
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    # reflection for a soft specular
    R = 2 * nz[..., None] * np.stack([nx, ny, nz], -1) - np.array([0, 0, 1.0])
    spec = np.clip((R * L).sum(-1), 0, 1) ** 40
    fres = (1 - nz) ** 3
    base = np.array(tint, np.float32) / 255
    col = base * (0.18 + 0.82 * lam[..., None]) + spec[..., None] * 1.1 + np.array(rim) / 255 * fres[..., None] * 0.9
    col = np.clip(col, 0, 1) * 255
    alpha = np.clip((1 - np.sqrt(d2)) * r * 1.5, 0, 1) * 255 * mask
    rgba = np.dstack([col, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA"), r


def block_sprite(w_m, h_m):
    w, h = int(w_m * PX_PER_M * S), int(h_m * PX_PER_M * S)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 0.55 + 0.45 * (1 - yy / h)  # lit from above
    col = np.array([200, 205, 216], np.float32)[None, None, :] * shade[..., None]
    col[: 3 * S, :, :] = 245  # top edge highlight
    im = Image.fromarray(col.astype(np.uint8), "RGB").convert("RGBA")
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=6 * S, fill=255)
    im.putalpha(m)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=6 * S, outline=(255, 85, 38, 150), width=S)
    return im


def draw_shadow(d, x_m, height_m, r_m):
    cx, fy = wx(x_m), FLOOR_Y * S
    k = max(0.0, 1 - height_m / 3.2)
    rw, rh = r_m * PX_PER_M * S * (0.6 + 0.6 * k), 5 * S * (0.5 + 0.5 * k)
    for i in range(5, 0, -1):
        a = int(26 * k * (6 - i) / 5)
        d.ellipse([cx - rw * i / 3, fy - rh * i / 3, cx + rw * i / 3, fy + rh * i / 3], fill=(0, 0, 0, a))


# ---------------------------------------------------------------- HUD
def text(d, xy, s, f, fill):
    d.text((xy[0] * S, xy[1] * S), s, font=f, fill=fill)


def pill(d, x, y, label, color, filled):
    tw = d.textlength(label, font=F_SMALL_B) / S
    box = [x * S, y * S, (x + tw + 18) * S, (y + 20) * S]
    if filled:
        d.rounded_rectangle(box, radius=10 * S, fill=color + (235,))
        d.text(((x + 9) * S, (y + 4) * S), label, font=F_SMALL_B, fill=(5, 5, 7))
    else:
        d.rounded_rectangle(box, radius=10 * S, outline=color + (220,), width=S, fill=(0, 0, 0, 150))
        d.text(((x + 9) * S, (y + 4) * S), label, font=F_SMALL_B, fill=color)
    return tw + 18


def chart(d, x, y, w, h, series, t_now, y_range, label, ref=None):
    """series: list of (t, v). ref: list of (t, v) dashed reference."""
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=8 * S, fill=(10, 11, 14, 215), outline=(255, 255, 255, 30), width=S)
    text(d, (x + 8, y + 5), label, F_TINY, MUTED)
    lo, hi = y_range
    px0, px1, py0, py1 = x + 8, x + w - 8, y + 20, y + h - 7

    def P(t, v):
        return ((px0 + (px1 - px0) * t / DUR) * S, (py1 - (py1 - py0) * (v - lo) / (hi - lo)) * S)

    if lo < 0 < hi:
        d.line([P(0, 0), P(DUR, 0)], fill=(255, 255, 255, 30), width=S)
    if ref:
        for i in range(0, len(ref) - 1, 2):
            d.line([P(*ref[i]), P(*ref[i + 1])], fill=(255, 255, 255, 110), width=S)
    pts = [P(t, v) for t, v in series if t <= t_now + 1e-9]
    if len(pts) > 1:
        d.line(pts, fill=series_color[0] + (255,), width=2 * S, joint="curve")
    if pts:
        cx, cy = pts[-1]
        d.ellipse([cx - 3 * S, cy - 3 * S, cx + 3 * S, cy + 3 * S], fill=series_color[0] + (255,))


series_color = [TEXT]


def frame_hud(d, idx, title, after, lines, verdict):
    text(d, (16, 14), idx, F_SMALL, FAINT)
    text(d, (16 + 30, 14), title, F_SMALL_B, TEXT)
    if after:
        pill(d, 16, 34, "AFTER · FERMI EVALS", ORANGE, False)
    else:
        pill(d, 16, 34, "BEFORE · RAW OUTPUT", (170, 176, 188), False)
    # metrics, bottom-left (two lines)
    y = H - 44
    for k, (lab, val, col) in enumerate(lines):
        text(d, (16, y + k * 16), lab, F_SMALL, MUTED)
        lw = d.textlength(lab, font=F_SMALL) / S
        text(d, (16 + lw + 6, y + k * 16), val, F_SMALL_B, col)
    # verdict bottom-right
    ok, label = verdict
    col = TEAL if ok else RED
    tw = d.textlength(label, font=F_SMALL_B) / S
    pill(d, W - 16 - tw - 18, H - 36, label, col, True)


def dashed(seg_pts, dash=0.06):
    """Turn a polyline in (t, v) into dash pairs for chart()."""
    out = []
    for (t0, v0), (t1, v1) in zip(seg_pts, seg_pts[1:]):
        n = max(1, int((t1 - t0) / dash))
        for k in range(0, n, 2):
            a, b = k / n, min(1, (k + 1) / n)
            out += [(t0 + (t1 - t0) * a, v0 + (v1 - v0) * a), (t0 + (t1 - t0) * b, v0 + (v1 - v0) * b)]
    return out


def save_gif(frames, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    small = [f.resize((W, H), Image.LANCZOS) for f in frames]
    # one shared palette built from a strip of sample frames
    sample = Image.new("RGB", (W, H * 4))
    for k, i in enumerate(np.linspace(0, len(small) - 1, 4).astype(int)):
        sample.paste(small[i], (0, H * k))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in small]
    path = os.path.join(OUT_DIR, name)
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=True, disposal=1)
    print(f"{name}: {os.path.getsize(path) // 1024} KB")


def new_frame():
    im = BG.copy().convert("RGBA")
    return im, ImageDraw.Draw(im, "RGBA")


def draw_trail(d, pts, r=2.2):
    n = len(pts)
    for k, (x, y) in enumerate(pts):
        a = int(150 * (k + 1) / n)
        cx, cy = wx(x), wy(y)
        d.ellipse([cx - r * S, cy - r * S, cx + r * S, cy + r * S], fill=(245, 247, 250, a))


# ================================================================ GRAVITY
G = 9.81
BALL_R = 0.22


def gravity_after(t):
    """Projectile with restitution 0.7. Returns x, y, vy."""
    x = 0.6 + 1.9 * t
    y0, vy0, e = BALL_R + 0.08, 6.0, 0.7
    tt = t
    while True:
        # time to hit floor (centre at BALL_R)
        disc = vy0 * vy0 + 2 * G * (y0 - BALL_R)
        tf = (vy0 + math.sqrt(disc)) / G
        if tt <= tf:
            return x, y0 + vy0 * tt - 0.5 * G * tt * tt, vy0 - G * tt
        tt -= tf
        vy0 = e * math.sqrt(disc)
        y0 = BALL_R


def gravity_before(t):
    """Floats, falls at constant speed, then bounces higher than it fell."""
    x = 0.6 + 1.9 * t
    top, top2 = 2.13, 2.6
    if t < 0.7:
        y, vy = BALL_R + 0.08 + (top - BALL_R - 0.08) * t / 0.7, (top - BALL_R - 0.08) / 0.7
    elif t < 1.5:
        y, vy = top + 0.04 * math.sin((t - 0.7) * 9), 0.36 * math.cos((t - 0.7) * 9)
    elif t < 2.05:
        v = (top - BALL_R) / 0.55
        y, vy = top - v * (t - 1.5), -v
    else:
        v = (top2 - BALL_R) / 0.95
        y, vy = BALL_R + v * (t - 2.05), v
    return x, y, vy


def render_gravity(after):
    set_floor(292)
    fn = gravity_after if after else gravity_before
    sprite, r = sphere_sprite(BALL_R, (210, 214, 224), ORANGE)
    ts = [i / FPS for i in range(NF)]
    traj = [fn(t) for t in ts]
    series = [(t, s[2]) for t, s in zip(ts, traj)]
    # dashed reference: free-fall slope -g from launch speed
    ref = dashed([(0, 6.0), (6.0 / G + 0.6, 6.0 - G * (6.0 / G + 0.6))])
    if after:
        bounce_loss = 1 - 0.7 ** 2
        lines = [("fitted g", "9.81 m/s²  (ref 9.81)", TEAL), ("bounce energy", f"−{bounce_loss * 100:.0f}%  (e = 0.70)", TEAL)]
        verdict = (True, "✓ PASS")
    else:
        gain = (2.6 - BALL_R) / (2.13 - BALL_R) - 1
        lines = [("fitted g", "0.0 m/s² mid-air  (ref 9.81)", RED), ("bounce energy", f"+{gain * 100:.0f}%  (gains energy)", RED)]
        verdict = (False, "✗ FAIL · 2 violations")
    frames = []
    for i, t in enumerate(ts):
        im, d = new_frame()
        x, y, _ = traj[i]
        draw_trail(d, [(s[0], s[1]) for s in traj[max(0, i - 36): i + 1: 2]])
        draw_shadow(d, x, y - BALL_R, BALL_R)
        im.alpha_composite(sprite, (int(wx(x) - r - 1), int(wy(y) - r - 1)))
        series_color[0] = TEAL if after else RED
        chart(d, W - 186, 14, 170, 84, series, t, (-7.5, 7.5), "vertical velocity v_y(t)", ref)
        frame_hud(d, "01", "GRAVITY", after, lines, verdict)
        frames.append(im.convert("RGB"))
    return frames


# ================================================================ MOMENTUM
MA, MB, RA, RB = 1.0, 3.0, 0.24, 0.36
XA0, VA0, XB0 = 1.0, 2.8, 4.0
T_HIT = (XB0 - RA - RB - XA0) / VA0


def momentum_after(t):
    if t < T_HIT:
        return XA0 + VA0 * t, XB0, VA0, 0.0
    va = (MA - MB) / (MA + MB) * VA0
    vb = 2 * MA / (MA + MB) * VA0
    xa_hit = XB0 - RA - RB
    return xa_hit + va * (t - T_HIT), XB0 + vb * (t - T_HIT), va, vb


OVERLAP = 0.15
VB_BAD = 1.9


def momentum_before(t):
    t_pen = T_HIT + OVERLAP / VA0
    if t < t_pen:
        return XA0 + VA0 * t, XB0, VA0, 0.0
    xa = XB0 - RA - RB + OVERLAP
    return xa, XB0 + VB_BAD * (t - t_pen), 0.0, VB_BAD


def render_momentum(after):
    set_floor(262)
    fn = momentum_after if after else momentum_before
    spA, rA = sphere_sprite(RA, (255, 170, 140), ORANGE)
    spB, rB = sphere_sprite(RB, (160, 200, 240), BLUE)
    ts = [i / FPS for i in range(NF)]
    traj = [fn(t) for t in ts]
    series = [(t, MA * s[2] + MB * s[3]) for t, s in zip(ts, traj)]
    p0 = MA * VA0
    ref = dashed([(0, p0), (DUR, p0)])
    if after:
        lines = [("Σp", f"{p0:.2f} → {p0:.2f} kg·m/s  (Δ 0.0%)", TEAL), ("contact", "no overlap · KE conserved", TEAL)]
        verdict = (True, "✓ PASS")
    else:
        p1 = MB * VB_BAD
        lines = [("Σp", f"{p0:.2f} → {p1:.2f} kg·m/s  (+{(p1 / p0 - 1) * 100:.0f}%)", RED), ("contact", f"bodies overlap {OVERLAP:.2f} m", RED)]
        verdict = (False, "✗ FAIL · 2 violations")
    frames = []
    for i, t in enumerate(ts):
        im, d = new_frame()
        xa, xb, _, _ = traj[i]
        draw_shadow(d, xa, 0, RA)
        draw_shadow(d, xb, 0, RB)
        im.alpha_composite(spB, (int(wx(xb) - rB - 1), int(wy(RB) - rB - 1)))
        im.alpha_composite(spA, (int(wx(xa) - rA - 1), int(wy(RA) - rA - 1)))
        for x, rr, m in ((xa, RA, "1 kg"), (xb, RB, "3 kg")):
            tw = d.textlength(m, font=F_TINY) / S
            text(d, (wx(x) / S - tw / 2, wy(2 * rr) / S - 18), m, F_TINY, MUTED)
        series_color[0] = TEAL if after else RED
        chart(d, W - 186, 14, 170, 84, series, t, (0, 7), "total momentum Σp(t)", ref)
        frame_hud(d, "02", "CONSERVATION OF MOMENTUM", after, lines, verdict)
        frames.append(im.convert("RGB"))
    return frames


# ================================================================ FRICTION
MU, V0, X0 = 0.45, 4.2, 0.4
BW, BH = 0.8, 0.5
DEC = MU * G
T_STOP = V0 / DEC
D_STOP = V0 * V0 / (2 * DEC)


def friction_after(t):
    if t < T_STOP:
        return X0 + V0 * t - 0.5 * DEC * t * t, V0 - DEC * t
    return X0 + D_STOP, 0.0


def friction_before(t):
    """Under-decelerates, then speeds up again with no force acting."""
    if t < 1.2:
        v = V0 - (V0 - 1.6) * t / 1.2
        x = X0 + (V0 + v) / 2 * t
    elif t < 2.2:
        x1 = X0 + (V0 + 1.6) / 2 * 1.2
        v = 1.6 + 0.8 * (t - 1.2)
        x = x1 + (1.6 + v) / 2 * (t - 1.2)
    else:
        x2 = X0 + (V0 + 1.6) / 2 * 1.2 + (1.6 + 2.4) / 2 * 1.0
        v = 2.4
        x = x2 + 2.4 * (t - 2.2)
    return x, v


def render_friction(after):
    set_floor(258)
    fn = friction_after if after else friction_before
    sprite = block_sprite(BW, BH)
    ts = [i / FPS for i in range(NF)]
    traj = [fn(t) for t in ts]
    series = [(t, s[1]) for t, s in zip(ts, traj)]
    ref = dashed([(0, V0), (T_STOP, 0), (DUR, 0)])
    if after:
        lines = [("deceleration", f"{DEC:.2f} m/s² = μg  (μ = {MU:.2f})", TEAL), ("stopping distance", f"{D_STOP:.2f} m  (ref {D_STOP:.2f})", TEAL)]
        verdict = (True, "✓ PASS")
    else:
        lines = [("deceleration", "2.17 m/s², then −0.80 (speeds up)", RED), ("friction μ_eff", "0.22 → −0.08  (negative)", RED)]
        verdict = (False, "✗ FAIL · 2 violations")
    frames = []
    for i, t in enumerate(ts):
        im, d = new_frame()
        # rough surface band + metre ruler
        fy = FLOOR_Y * S
        for k in range(0, W * S, 6 * S):
            d.line([(k, fy + 2 * S), (k + 4 * S, fy + 6 * S)], fill=(255, 255, 255, 26), width=S)
        for m in range(0, 8):
            x = wx(X0 + m)
            d.line([(x, fy + 8 * S), (x, fy + 14 * S)], fill=(255, 255, 255, 70), width=S)
            text(d, (x / S - 3, FLOOR_Y + 16), f"{m}m", F_TINY, FAINT)
        if after:
            xs = wx(X0 + D_STOP + BW / 2)
            d.line([(xs, fy - 4 * S), (xs, fy + 14 * S)], fill=TEAL + (160,), width=S)
        x, _ = traj[i]
        left = wx(x)
        d.rectangle([left + 6 * S, fy - 2 * S, left + BW * PX_PER_M * S - 6 * S, fy + 2 * S], fill=(0, 0, 0, 120))
        im.alpha_composite(sprite, (int(left), int(wy(BH))))
        series_color[0] = TEAL if after else RED
        chart(d, W - 186, 14, 170, 84, series, t, (-0.4, 4.8), "speed v(t)", ref)
        frame_hud(d, "03", "FRICTION", after, lines, verdict)
        frames.append(im.convert("RGB"))
    return frames


def main():
    for name, fn in (("gravity", render_gravity), ("momentum", render_momentum), ("friction", render_friction)):
        save_gif(fn(False), f"{name}-before.gif")
        save_gif(fn(True), f"{name}-after.gif")


if __name__ == "__main__":
    main()

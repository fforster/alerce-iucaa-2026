"""Draw figures/database_schematic.png: what the ALeRCE multisurvey database stores about one object.

    python tools/make_schematic.py

Everything is synthetic and schematic (no network access needed).
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import Circle, ConnectionPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "database_schematic.png"

# Helvetica (or its metric clones TeX Gyre Heros / Nimbus Sans when Helvetica is not installed)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "Arial"]

# Shades of the ALeRCE logo blue
LOGO = "#2E3192"


def shade(t):
    """t < 0: darker than the logo blue (towards black); t > 0: lighter (towards white)."""
    c = np.array(to_rgb(LOGO))
    return to_hex(c * (1 + t) if t < 0 else c + (1 - c) * t)


INK, INK2, MUTED = shade(-0.45), shade(0.25), shade(0.6)   # titles / secondary text / thin lines
ACCENT, ACCENT_LIGHT = LOGO, shade(0.72)
FAMILY = {  # pale tints of the logo blue, one per table family
    "object": shade(0.90),
    "lightcurve": shade(0.95),
    "derived": shade(0.92),
    "xmatch": shade(0.93),
    "lut": shade(0.96),
}
BORDER = shade(0.78)
STAMP_CMAP = LinearSegmentedColormap.from_list("alerce", [shade(-0.7), LOGO, shade(0.55), "white"])
# ALeRCE band colours (Okabe-Ito) keep their meaning in the light curve
BAND = {"g": "#009E73", "r": "#D55E00"}
MONO = {}

rng = np.random.default_rng(7)
fig = plt.figure(figsize=(16, 9.6), dpi=130)
fig.patch.set_facecolor("white")


def card(rect, family, title, tables):
    """Rounded background box with a title and the table names that hold this information."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, fc=FAMILY[family], ec=BORDER, lw=1))
    ax.text(0.03, 0.95, title, transform=ax.transAxes, va="top", fontsize=13, weight="bold", color=INK)
    ax.text(0.97, 0.95, tables, transform=ax.transAxes, va="top", ha="right", fontsize=9.5, color=INK2, **MONO)
    return ax


# ---------------------------------------------------------------- 1. the object
ax = card([0.015, 0.50, 0.215, 0.44], "object", "1 · the object", "object\nztf_object · lsst_dia_object")
sky = fig.add_axes([0.04, 0.70, 0.16, 0.16])
sky.set_xlim(-1, 1); sky.set_ylim(-1, 1); sky.set_aspect("equal"); sky.set_axis_off()
for _ in range(14):
    x, y = rng.uniform(-0.95, 0.95), rng.uniform(-0.35, 0.95)
    if np.hypot(x, y) > 0.4:
        sky.plot(x, y, "o", ms=rng.uniform(1.5, 4), color=MUTED)
sky.add_patch(Circle((0, 0), 0.32, fc="none", ec=INK, lw=1.2))
sky.plot(0, 0, "*", ms=14, color=ACCENT, mec=INK, mew=0.6)
sky.text(0, -0.55, "one source on the sky", ha="center", fontsize=9, color=INK2)
lines = [("oid", "36029005852662990"), ("sid", "0 = ZTF   (1 = LSST)"), ("meanra, meandec", "149.9287, +0.4184"),
         ("firstmjd, lastmjd", "61121.3, 61185.2"), ("n_det, n_forced", "39, 36")]
for k, (key, val) in enumerate(lines):
    y = 0.39 - 0.072 * k
    ax.text(0.05, y, key, transform=ax.transAxes, fontsize=9, color=INK2, **MONO)
    ax.text(0.05, y - 0.034, val, transform=ax.transAxes, fontsize=9.5, color=INK, **MONO)

# ---------------------------------------------------------------- 2. the light curve
ax = card([0.245, 0.30, 0.47, 0.64], "lightcurve", "2 · the light curve: one row per observation",
          "detection + ztf_/lsst_detection\nforced_photometry + ztf_/lsst_forced_photometry\nztf_non_detection")
lc = fig.add_axes([0.29, 0.44, 0.40, 0.37])
t0 = 0.0


def model(t, peak, rise, fall):
    f = np.where(t > t0, (1 - np.exp(-(t - t0) / rise)) * np.exp(-(t - t0) / fall), 0)
    return peak * f / np.max(f)


t_obs = np.sort(rng.uniform(-35, 70, 70))
bands = np.where(rng.random(t_obs.size) < 0.5, "g", "r")
params = {"g": (1.0, 4, 22), "r": (0.85, 5, 35)}
err = 0.045
for b in "gr":
    tt = np.linspace(-40, 75, 400)
    lc.plot(tt, model(tt, *params[b]), color=BAND[b], lw=1, alpha=0.35, zorder=1)
for b, zo in (("r", 5), ("g", 6)):
    sel = bands == b
    t = t_obs[sel]
    f = model(t, *params[b]) + rng.normal(0, err, t.size)
    det = f > 5 * err
    lc.errorbar(t[~det & (t > -5)], f[~det & (t > -5)], err, fmt="o", ms=6, mfc="white", color=BAND[b],
                elinewidth=0.8, zorder=zo)
    lc.errorbar(t[t <= -5], f[t <= -5], err, fmt="o", ms=6, mfc="white", color=BAND[b], elinewidth=0.8,
                zorder=zo, label=f"{b}: forced photometry")
    lc.errorbar(t[det], f[det], err, fmt="o", ms=6, color=BAND[b], elinewidth=0.8, zorder=zo + 0.5,
                label=f"{b}: detection (alert)")
    if b == "g":  # the detection whose stamps we show: the g detection closest to maximum
        k = np.argmin(np.abs(t[det] - 6))
        t_star, f_star = t[det][k], f[det][k]
    nd = t[(t < -2)][::3]
    lc.plot(nd, np.full(nd.size, 5 * err + 0.04), "v", ms=7, color=BAND[b], alpha=0.45, zorder=zo,
            label=f"{b}: non-detection (ZTF only)")
lc.axhline(0, color=MUTED, lw=0.6)
lc.set_xlim(-40, 75)
lc.set_ylim(-0.2, 1.55)
lc.set_xlabel("time  [MJD]", color=INK2)
lc.set_ylabel("difference flux", color=INK2)
lc.tick_params(colors=MUTED, labelsize=8)
for s in ("top", "right"):
    lc.spines[s].set_visible(False)
for s in ("left", "bottom"):
    lc.spines[s].set_color(MUTED)
handles, labels = lc.get_legend_handles_labels()
order = [labels.index(l) for l in ("g: detection (alert)", "g: forced photometry", "g: non-detection (ZTF only)",
                                   "r: detection (alert)", "r: forced photometry", "r: non-detection (ZTF only)")]
lc.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8.5, ncol=2, frameon=False,
          loc="upper right", labelcolor=INK)
lc.annotate("before the explosion:\nforced photometry ≈ 0", xy=(-22, 0.0), xytext=(-37, 0.45), fontsize=9,
            color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
# the detection that will get stamps
lc.plot(t_star, f_star, "o", ms=13, mfc="none", mec=INK, mew=1.2, zorder=10)
ax.text(0.04, 0.04, "each point = one row:\n"
        "oid, sid, measurement_id, mjd, band, ra, dec\n"
        "+ survey columns: ZTF mag / LSST flux [nJy]",
        transform=ax.transAxes, fontsize=9, color=INK2, **MONO)

# ---------------------------------------------------------------- 3. stamps
ax = card([0.245, 0.03, 0.47, 0.255], "lightcurve", "3 · image stamps of one detection",
          "stamps API  (not in TAP)")
yy, xx = np.mgrid[-15:16, -15:16]
galaxy = 0.6 * np.exp(-((xx * 0.9 + yy * 0.45) ** 2 / 60 + (yy * 0.9 - xx * 0.45) ** 2 / 18))
sn = np.exp(-((xx - 2) ** 2 + (yy - 1) ** 2) / 5)
noise = lambda: rng.normal(0, 0.03, xx.shape)
stamps = [("science", galaxy + sn + noise()), ("template", galaxy + noise()), ("difference", sn + noise())]
for k, (name, img) in enumerate(stamps):
    s_ax = fig.add_axes([0.275 + k * 0.105, 0.055, 0.085, 0.15])
    s_ax.imshow(img, cmap=STAMP_CMAP, origin="lower", vmin=-0.05, vmax=1.1)
    s_ax.set_xticks([]); s_ax.set_yticks([])
    for sp in s_ax.spines.values():
        sp.set_color(MUTED)
    s_ax.set_title(name, fontsize=9.5, color=INK)
    if k < 2:
        fig.text(0.275 + k * 0.105 + 0.093, 0.13, "−" if k == 0 else "=", fontsize=18, color=INK2,
                 ha="center", va="center")
fig.text(0.597, 0.13, "science − template\n= what changed", fontsize=9.5, color=INK2, va="center")
con = ConnectionPatch(xyA=(t_star, f_star), coordsA=lc.transData, xyB=(0.275 + 0.105 * 2 + 0.042, 0.205),
                      coordsB=fig.transFigure, arrowstyle="->", color=INK, lw=1, connectionstyle="arc3,rad=-0.2")
fig.add_artist(con)

# ---------------------------------------------------------------- 4. features
ax = card([0.73, 0.645, 0.255, 0.295], "derived", "4 · features", "feature\nfeature_name_lut")
rows = [("feature_name", "band", "value"), ("SPM_tau_rise", "g", "4.1"), ("SPM_tau_fall", "r", "34.8"),
        ("Multiband_period", "gr", "0.4889"), ("g-r_max", "", "0.40"), ("W1-W2", "", "0.12"),
        ("…", "", "~130 / object")]
for k, (a, b, c) in enumerate(rows):
    y = 0.70 - 0.095 * k
    w = "bold" if k == 0 else "normal"
    col = INK2 if k == 0 else INK
    ax.text(0.05, y, a, transform=ax.transAxes, fontsize=9.5, color=col, weight=w, **MONO)
    ax.text(0.60, y, b, transform=ax.transAxes, fontsize=9.5, color=col, weight=w, **MONO)
    ax.text(0.95, y, c, transform=ax.transAxes, fontsize=9.5, color=col, weight=w, ha="right", **MONO)
ax.text(0.05, 0.035, "numbers that summarise the light curve", transform=ax.transAxes, fontsize=9, color=INK2)

# ---------------------------------------------------------------- 5. probabilities
ax = card([0.73, 0.30, 0.255, 0.33], "derived", "5 · classifications", "probability\nclassifier · taxonomy")
pr = fig.add_axes([0.80, 0.355, 0.165, 0.175])
classes = ["SESN", "SNIa", "SNII", "SNIIn", "SLSN"]
probs = [0.45, 0.22, 0.19, 0.09, 0.05]
pr.barh(classes[::-1], probs[::-1], color=[ACCENT_LIGHT] * 4 + [ACCENT], height=0.6)
for k, p in enumerate(probs[::-1]):
    pr.text(p + 0.01, k, f"{p:.2f}", va="center", fontsize=8.5, color=INK)
pr.set_xlim(0, 0.62)
pr.set_facecolor("none")
pr.tick_params(colors=INK, labelsize=8.5, length=0)
pr.set_xticks([])
for sp in pr.spines.values():
    sp.set_visible(False)
ax.text(0.05, 0.80, "e.g. the light-curve classifier, all its classes;\nranking = 1 → most probable",
        transform=ax.transAxes, fontsize=9, color=INK2, va="top")

# ---------------------------------------------------------------- 6. crossmatches
ax = card([0.015, 0.155, 0.215, 0.33], "xmatch", "6 · crossmatches", "xmatch\nallwise · gaiadr3_source")
xm = fig.add_axes([0.035, 0.205, 0.10, 0.17])
xm.set_xlim(-1, 1); xm.set_ylim(-1, 1); xm.set_aspect("equal"); xm.set_axis_off()
xm.add_patch(Circle((0, 0), 0.8, fc="none", ec=MUTED, lw=0.8, ls="--"))
xm.plot(0, 0, "*", ms=12, color=ACCENT, mec=INK, mew=0.6)
xm.plot(0.25, 0.15, "s", ms=8, mfc="none", mec=INK, mew=1.1)
xm.plot(-0.2, -0.3, "D", ms=7, mfc="none", mec=INK2, mew=1.1)
xm.annotate("", xy=(0.25, 0.15), xytext=(0, 0), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax.text(0.55, 0.62, "□ AllWISE\n   W1−W2 = 1.3\n\n◇ Gaia DR3\n   parallax ≈ 0", transform=ax.transAxes,
        fontsize=9, color=INK, va="top")
ax.text(0.05, 0.05, "dist = 0.4″ to the nearest source", transform=ax.transAxes, fontsize=9, color=INK2)

# ---------------------------------------------------------------- 7. magstat (small)
ax = card([0.015, 0.03, 0.215, 0.11], "derived", "7 · per-band statistics", "")
ax.text(0.05, 0.42, "magstat  (ZTF only, for now)", transform=ax.transAxes, fontsize=9.5, color=INK2, **MONO)
ax.text(0.05, 0.14, "magmin, magmax, magmean, dmdt_first …", transform=ax.transAxes, fontsize=9, color=INK,
        **MONO)

# ---------------------------------------------------------------- 8. lookup tables
ax = card([0.73, 0.03, 0.255, 0.255], "lut", "8 · look-up tables", "decode the integers")
luts = [("sid_lut", "0 → ZTF, 1 → LSST"), ("band", "(sid, 1) → g"), ("feature_name_lut", "(sid, 27) → period"),
        ("classifier / taxonomy", "(5, 18) → SNIa"), ("catalog_id_lut", "0 → AllWISE")]
for k, (a, b) in enumerate(luts):
    y = 0.70 - 0.13 * k
    ax.text(0.05, y, a, transform=ax.transAxes, fontsize=9.5, color=INK, **MONO)
    ax.text(0.95, y, b, transform=ax.transAxes, fontsize=9, color=INK2, ha="right")

fig.suptitle("What ALeRCE stores about one object  (the same structure for ZTF and LSST)",
             x=0.015, y=0.985, ha="left", fontsize=16, weight="bold", color=INK)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)

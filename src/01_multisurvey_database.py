# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/01_multisurvey_database.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 01 · The ALeRCE multisurvey database
#
# **ALeRCE multisurvey tutorial — IUCAA 2026** · ~45 min
#
# Since October 2026 ALeRCE stores **ZTF and LSST in one PostgreSQL database** (schema `multisurvey`), published
# through TAP as the schema **`alerce_tap`**. In this notebook we visit **every one of its 31 tables**, learn the
# rules to join them, and write the small helper functions that we will reuse in the science notebooks.
#
# <img src="https://raw.githubusercontent.com/fforster/alerce-iucaa-2026/main/figures/database_schematic.png" width="100%" alt="What ALeRCE stores about one object"/>
#
# The map above (from notebook 00) is our guide: each numbered box is one or more tables, and the sections below open
# them one by one.
#
# **Contents**
# 1. The identity model: `oid`, `sid`, `tid`
# 2. The map of the database
# 3. Look-up tables (LUTs) and a `decode()` helper
# 4. Objects: `object`, `ztf_object`, `lsst_dia_object`
# 5. Light curves: `detection`, `forced_photometry`, `ztf_non_detection` and their survey extensions
# 6. ZTF auxiliary tables: `ztf_ps1`, `ztf_gaia`, `ztf_ss`, `ztf_dataquality`, `ztf_reference`
# 7. LSST solar-system tables: `lsst_ss_object`, `lsst_ss_detection`, `lsst_mpc_orbits`
# 8. Derived products: `magstat`, `feature`, `probability`
# 9. Crossmatches: `xmatch`, `allwise`, `gaiadr3_source`
# 10. Rules of thumb, and the `alerce_ms` library
#
# The structure follows chapter 7 of the *ALeRCE book* (ALeRCE's internal design document).
#
# 🔎 Keep the **[alerce_tap schema explorer](https://claude.ai/artifact/JEpEyijsTpD1y83ut7Zia7)** open in another tab
# while you work: an interactive guide to all 31 tables and their columns, primary keys, partitions and indexes, with
# example values, the relations between tables, search, and science query examples for LSST and ZTF.

# %%
import sys

if "google.colab" in sys.modules:
    !pip -q install pyvo astroquery
    !wget -q -N https://raw.githubusercontent.com/fforster/alerce-iucaa-2026/main/alerce_ms.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvo

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

tap = pyvo.dal.TAPService("https://tap.alerce.online/tap")


def tap_query(adql, maxrec=1_000_000):
    """Run an ADQL query and return a pandas DataFrame."""
    return tap.search(adql, maxrec=maxrec).to_table().to_pandas()

# %% [markdown]
# > **Be gentle with the server.** ALeRCE services are free, shared and rate-limited per IP address. When a whole
# > classroom shares one IP, avoid running loops of hundreds of queries; prefer one query that returns many rows.
# > The `alerce_ms` library we import at the end caches every result on disk and retries politely.

# %% [markdown]
# ## 1. The identity model
#
# Every row about an astronomical object carries **two** keys:
#
# | Key | Type | Meaning |
# |---|---|---|
# | `sid` | int2 | **survey id**: 0 = ZTF, 1 = LSST DIA object, 2 = LSST solar-system object |
# | `oid` | int8 | object id **within that survey** |
# | `tid` | int2 | telescope id: 0 = ZTF (Palomar), 1 = Rubin |
#
# > 🔑 **Rule 1: an object is the pair `(oid, sid)`. Always join on both columns.** The same integer could
# > in principle be an `oid` in two surveys.
#
# > 🔑 **Rule 2: there is no cross-survey association.** A star observed by ZTF and by LSST is *two objects*. To link
# > them you do a positional crossmatch (notebook 02).
#
# **LSST** `oid` is simply Rubin's `diaObjectId`, e.g. `314003014107006318`.
#
# **ZTF** names like `ZTF26aaonmha` are converted to integers. The name is a two-digit year plus seven lower-case
# letters, i.e. a number in base 26. ALeRCE adds a *survey prefix* (ZTF = 1) in the top bits:
# `oid = (1 << 55) + year·26⁷ + letters(base 26)`. Let us write the encoder and decoder:

# %%
def ztf_name_to_oid(name):
    """'ZTF26aaonmha' -> integer oid used in alerce_tap."""
    year, letters = int(name[3:5]), name[5:12]
    seq = sum((ord(c) - ord("a")) * 26 ** (6 - i) for i, c in enumerate(letters))
    return (1 << 55) + year * 26 ** 7 + seq


def oid_to_ztf_name(oid):
    """Inverse of ztf_name_to_oid."""
    value = int(oid) & ((1 << 55) - 1)          # remove the survey prefix
    year, seq = (value // 26 ** 7) % 100, value % 26 ** 7
    letters = ""
    for i in range(6, -1, -1):
        letters += chr(seq // 26 ** i + ord("a"))
        seq %= 26 ** i
    return f"ZTF{year:02d}{letters}"


oid = ztf_name_to_oid("ZTF26aaonmha")
print(oid, "->", oid_to_ztf_name(oid))

# %% [markdown]
# > ⚠️ The survey *prefix* inside the ZTF integer (ZTF = 1, ATLAS = 2, LSST = 3, …) is **not** the same numbering
# > as `sid`. Never mix them.
# >
# > ⚠️ LSST ids are 64-bit integers: they do not fit in a JavaScript/JSON double. In Python they are fine, but never
# > store them as `float`.

# %% [markdown]
# ## 2. The map of the database
#
# The TAP service describes itself in `TAP_SCHEMA`. Let us list the tables and their descriptions:

# %%
tables = tap_query("""
    SELECT table_name, description
    FROM TAP_SCHEMA.tables
    WHERE schema_name = 'alerce_tap'
""")
tables["table_name"] = tables.table_name.str.replace("alerce_tap.", "")
tables["description"] = tables.description.str.slice(0, 110)
tables

# %% [markdown]
# They fall into a few families:
#
# ```
#                    shared (all surveys)          ZTF extension            LSST extension
#  objects        :  object                  ->    ztf_object               lsst_dia_object, lsst_ss_object
#  light curves   :  detection               ->    ztf_detection            lsst_detection, lsst_ss_detection
#                    forced_photometry       ->    ztf_forced_photometry    lsst_forced_photometry
#                                                  ztf_non_detection        (none)
#  ZTF extras     :                                ztf_ps1, ztf_gaia, ztf_ss, ztf_dataquality, ztf_reference
#  LSST orbits    :                                                         lsst_mpc_orbits
#  derived        :  magstat, feature, probability, xmatch
#  classifiers    :  classifier, taxonomy
#  catalogs       :  allwise, gaiadr3_source
#  look-up tables :  sid_lut, band, feature_name_lut, feature_version_lut, catalog_id_lut
# ```
#
# The **shared + extension** design is the key idea: every light-curve point is *one row* in a shared table
# (`detection`: `oid, sid, measurement_id, mjd, ra, dec, band`) plus *one row* with the survey's native columns in its
# extension table (`ztf_detection` has magnitudes, `lsst_detection` has fluxes and ~100 Rubin columns).
#
# You can list the columns of any table with `TAP_SCHEMA.columns` (or browse them, with the indexed columns marked, in
# the [schema explorer](https://claude.ai/artifact/JEpEyijsTpD1y83ut7Zia7)):

# %%
def tap_columns(table):
    return tap_query(f"""
        SELECT column_name, datatype, unit, description
        FROM TAP_SCHEMA.columns
        WHERE table_name = 'alerce_tap.{table}'
    """)


tap_columns("object")

# %% [markdown]
# ## 3. Look-up tables
#
# Large tables store small integers (`band = 2`, `feature_id = 27`, `class_id = 18`) instead of strings. The meaning of
# each integer lives in a **look-up table** (LUT). They are tiny, so we download them all once.
#
# ### 3.1 `sid_lut`: surveys

# %%
sid_lut = tap_query("SELECT * FROM alerce_tap.sid_lut")
sid_lut

# %% [markdown]
# ### 3.2 `band`: photometric bands
#
# Band numbers are **per survey**: always decode with `(sid, band)`. Note that ZTF has g, r, i (1, 2, 3, the old
# `fid`) and LSST has u, g, r, i, z, y, with `u` stored as 6 but displayed first (`band_order`).

# %%
band_lut = tap_query("SELECT sid, band, band_name, band_order FROM alerce_tap.band ORDER BY sid, band_order")
band_lut.pivot(index="sid", columns="band", values="band_name")

# %% [markdown]
# ### 3.3 `classifier` and `taxonomy`: classifiers and their classes
#
# `classifier.tid` tells which telescope the model is for. `class_id` values are numbered **per classifier**, so
# decode with `(classifier_id, class_id)`.

# %%
classifier_lut = tap_query("SELECT * FROM alerce_tap.classifier ORDER BY tid, classifier_id")
classifier_lut

# %%
taxonomy_lut = tap_query("SELECT classifier_id, class_id, class_name, taxonomy_order FROM alerce_tap.taxonomy")
taxonomy_lut.groupby("classifier_id").class_name.apply(", ".join).to_frame()

# %% [markdown]
# A new model, or a new version that must coexist with the old one, gets a **new `classifier_id`** (e.g.
# `stamp_classifier_rubin_beta_20260421`). Inside one `classifier_id` the database keeps only the latest prediction.
#
# ### 3.4 `feature_name_lut` and `feature_version_lut`: light-curve features
#
# ALeRCE computes ~130 features per object (colors, periods, variability statistics, supernova-model fits, …).
# Feature ids **differ between surveys**: decode with `(sid, feature_id)`.

# %%
feature_name_lut = tap_query("SELECT sid, feature_id, feature_name FROM alerce_tap.feature_name_lut")
print(feature_name_lut.groupby("sid").size())
feature_name_lut[feature_name_lut.feature_name.isin(["Multiband_period", "SPM_tau_rise", "GP_DRW_tau", "W1-W2"])]

# %%
feature_version_lut = tap_query("SELECT * FROM alerce_tap.feature_version_lut")
feature_version_lut

# %% [markdown]
# ### 3.5 `catalog_id_lut`: external catalogs used in `xmatch`

# %%
catalog_lut = tap_query("SELECT * FROM alerce_tap.catalog_id_lut")
catalog_lut

# %% [markdown]
# ### 3.6 A `decode()` helper
#
# Rather than remembering all these joins, we write one function that adds readable columns for every code it
# finds in a DataFrame. (You could also do these joins in ADQL; doing them in pandas keeps queries short.)

# %%
def band_label(sid, band):
    """Decode a band, including feature band combinations: 12 -> 'gr', 0 -> ''."""
    names = dict(zip(band_lut[band_lut.sid == sid].band, band_lut[band_lut.sid == sid].band_name))
    if band in names:
        return names[band]
    if band == 0:
        return ""
    return "".join(names.get(int(d), "?") for d in str(band))


def decode(df):
    out = df.copy()
    if "sid" in out:
        out = out.merge(sid_lut[["sid", "survey_name"]], on="sid", how="left")
        if "band" in out:
            out["band_name"] = [band_label(s, b) for s, b in zip(out.sid, out.band)]
        if "feature_id" in out:
            out = out.merge(feature_name_lut, on=["sid", "feature_id"], how="left")
    if "classifier_id" in out:
        out = out.merge(classifier_lut[["classifier_id", "classifier_name"]], on="classifier_id", how="left")
        if "class_id" in out:
            out = out.merge(taxonomy_lut[["classifier_id", "class_id", "class_name"]],
                            on=["classifier_id", "class_id"], how="left")
    if "catid" in out:
        out = out.merge(catalog_lut[["catid", "catalog_name"]], on="catid", how="left")
    return out

# %% [markdown]
# ## 4. Objects
#
# ### 4.1 `object`: one row per object, all surveys
#
# Mean position, first/last detection time, and number of detections (`n_det`), forced-photometry points
# (`n_forced`) and non-detections (`n_non_det`, ZTF only). Let us fetch our three COSMOS examples in ZTF:

# %%
names = ["ZTF17aabhbva", "ZTF26aaonmha", "ZTF20acyxpwi"]
oids = ", ".join(str(ztf_name_to_oid(n)) for n in names)

objects = tap_query(f"SELECT * FROM alerce_tap.object WHERE sid = 0 AND oid IN ({oids})")
objects["name"] = objects.oid.apply(oid_to_ztf_name)
objects[["name", "oid", "sid", "meanra", "meandec", "firstmjd", "lastmjd", "n_det", "n_forced", "n_non_det"]]

# %% [markdown]
# A **cone search** uses ADQL geometry functions. The radius is in **degrees**. This finds every object, in any
# survey, within 2″ of the supernova:

# %%
ra, dec = 149.928671, 0.418448
tap_query(f"""
    SELECT oid, sid, meanra, meandec, n_det, firstmjd,
           3600 * DISTANCE(POINT('ICRS', meanra, meandec), POINT('ICRS', {ra}, {dec})) AS sep_arcsec
    FROM alerce_tap.object
    WHERE 1 = CONTAINS(POINT('ICRS', meanra, meandec), CIRCLE('ICRS', {ra}, {dec}, 2.0 / 3600))
""").pipe(decode)

# %% [markdown]
# There it is: the ZTF object (`sid = 0`) and its LSST twin (`sid = 1`), 0.13″ apart.
#
# ### 4.2 `ztf_object` and `lsst_dia_object`: survey-specific object columns
#
# `ztf_object` adds ZTF quantities such as mean/max g−r colour, whether the source is `stellar`, whether the
# magnitudes could be `corrected` for the reference flux, and the alert history (`ndethist`).
# It has no `sid` column: it is ZTF by definition, so join on `oid` only.

# %%
tap_query(f"""
    SELECT o.oid, o.n_det, z.g_r_mean, z.g_r_max, z.stellar, z.corrected, z.ndethist, z.ncovhist
    FROM alerce_tap.object AS o
    JOIN alerce_tap.ztf_object AS z ON z.oid = o.oid
    WHERE o.sid = 0 AND o.oid IN ({oids})
""")

# %% [markdown]
# `lsst_dia_object` is Rubin's own *DiaObject* record (83 columns): per-band mean PSF fluxes on the difference image
# (`*_psffluxmean`), mean fluxes on the direct image (`*_sciencefluxmean`), forced-photometry means (`*_fpfluxmean`), etc.

# %%
tap_query("""
    SELECT oid, ra, dec, ndiasources,
           g_psffluxmean, r_psffluxmean, i_psffluxmean,
           g_sciencefluxmean, r_sciencefluxmean, i_sciencefluxmean
    FROM alerce_tap.lsst_dia_object
    WHERE oid = 314003014107006318
""")

# %% [markdown]
# ## 5. Light curves
#
# | | ZTF | LSST |
# |---|---|---|
# | brightness unit | magnitudes (`magpsf`, `mag`) | fluxes in nJy (`psfflux`, `scienceflux`) |
# | time | MJD (UTC) | MJD (TAI; 37 s ahead of UTC) |
# | detections | `detection` + `ztf_detection` | `detection` + `lsst_detection` |
# | forced photometry | `forced_photometry` + `ztf_forced_photometry` | `forced_photometry` + `lsst_forced_photometry` |
# | upper limits | `ztf_non_detection` | — (use forced photometry) |
#
# **Detections** come from alerts (a ≥5σ source on the difference image). **Forced photometry** measures the flux at
# the object position on every image, detected or not — it is how we see the pre-explosion baseline of a supernova.
#
# ### 5.1 LSST detections
#
# Join the shared table with its extension on `(oid, measurement_id, sid)`:

# %%
lsst_det = tap_query("""
    SELECT d.mjd, d.band, x.psfflux, x.psffluxerr, x.scienceflux, x.sciencefluxerr, x.reliability, x.snr
    FROM alerce_tap.detection AS d
    JOIN alerce_tap.lsst_detection AS x
      ON x.oid = d.oid AND x.measurement_id = d.measurement_id AND x.sid = d.sid
    WHERE d.oid = 314003014107006318 AND d.sid = 1
    ORDER BY d.mjd
""")
lsst_det.assign(sid=1).pipe(decode).head()

# %% [markdown]
# * `psfflux`: PSF flux on the **difference** image (science − template). Can be negative.
# * `scienceflux`: flux on the **direct** (science) image: source + host + everything else.
# * `reliability`: Rubin's real/bogus score (0–1).
#
# LSST forced photometry has the same structure (`forced_photometry` + `lsst_forced_photometry`): the flux measured
# at the diaObject position in visits sent with the alerts, whether or not there was a detection.

# %%
lsst_fp = tap_query("""
    SELECT f.mjd, f.band, x.psfflux, x.psffluxerr, x.scienceflux, x.visit, x.detector
    FROM alerce_tap.forced_photometry AS f
    JOIN alerce_tap.lsst_forced_photometry AS x
      ON x.oid = f.oid AND x.measurement_id = f.measurement_id AND x.sid = f.sid
    WHERE f.oid = 314003014107006318 AND f.sid = 1
    ORDER BY f.mjd
""")
print(len(lsst_det), "detections and", len(lsst_fp), "forced-photometry points")
lsst_fp.assign(sid=1).pipe(decode).head()

# %% [markdown]
# COSMOS is a Rubin *deep drilling field*, observed many times per night. The first rows above are from before the
# supernova: difference fluxes scatter around zero (±1000 nJy), while the direct-image `scienceflux` (~50 000 nJy in g)
# is the host galaxy. Some of those noisy early points even triggered low-significance "detections" — always check
# the signal-to-noise before trusting a single point.

# %% [markdown]
# ### 5.2 ZTF detections and forced photometry
#
# ZTF extension tables have no `sid` column, so join on `(oid, measurement_id)`. `magpsf` is the difference-image
# magnitude, `isdiffpos` tells if the difference is positive (+1) or negative (−1), and `magpsf_corr` is the
# apparent magnitude after adding back the reference flux (only when `corrected`). An error of **100** means
# "not available".

# %%
sn_oid = ztf_name_to_oid("ZTF26aaonmha")
ztf_det = tap_query(f"""
    SELECT d.mjd, d.band, x.magpsf, x.sigmapsf, x.isdiffpos, x.magpsf_corr, x.rb, x.drb
    FROM alerce_tap.detection AS d
    JOIN alerce_tap.ztf_detection AS x ON x.oid = d.oid AND x.measurement_id = d.measurement_id
    WHERE d.oid = {sn_oid} AND d.sid = 0
    ORDER BY d.mjd
""")
ztf_fp = tap_query(f"""
    SELECT f.mjd, f.band, x.mag, x.e_mag, x.isdiffpos, x.mag_corr, x.e_mag_corr
    FROM alerce_tap.forced_photometry AS f
    JOIN alerce_tap.ztf_forced_photometry AS x ON x.oid = f.oid AND x.measurement_id = f.measurement_id
    WHERE f.oid = {sn_oid} AND f.sid = 0
    ORDER BY f.mjd
""")
ztf_nd = tap_query(f"SELECT mjd, band, diffmaglim FROM alerce_tap.ztf_non_detection WHERE oid = {sn_oid}")
print(len(ztf_det), "detections,", len(ztf_fp), "forced-photometry points,", len(ztf_nd), "non-detections")
ztf_fp.head()

# %% [markdown]
# ### 5.3 Putting ZTF and LSST on the same scale
#
# To compare the two surveys we convert ZTF magnitudes into **signed difference fluxes in nJy**:
#
# $$F = \mathrm{isdiffpos}\times 10^{-0.4\,(m - 31.4)}\ \mathrm{nJy},\qquad \sigma_F = 0.4\ln 10\,|F|\,\sigma_m .$$
#
# For the plot we follow the ALeRCE conventions: band colours from the colour-blind-friendly **Okabe–Ito palette**
# ([Okabe & Ito 2008](https://jfly.uni-koeln.de/color/); recommended in *Nature Methods* by
# [Wong 2011](https://doi.org/10.1038/nmeth.1618)), the g band drawn on top, and the legend sorted by wavelength.

# %%
def mag_to_flux(mag):
    return 10 ** (-0.4 * (np.asarray(mag, float) - 31.4))


def ztf_diff_flux(df, mag="mag", err="e_mag"):
    df = df.copy()
    df.loc[df[err] >= 100, err] = np.nan
    df["flux"] = np.where(df.isdiffpos < 0, -1, 1) * mag_to_flux(df[mag])
    df["flux_err"] = 0.4 * np.log(10) * df.flux.abs() * df[err]
    return df


ztf_fp_flux = ztf_diff_flux(ztf_fp)
ztf_det_flux = ztf_diff_flux(ztf_det, "magpsf", "sigmapsf")

# ALeRCE plotting conventions: colour per band (same in ZTF and LSST) from the Okabe-Ito colour-blind-friendly
# palette (Okabe & Ito 2008; recommended in Nature Methods by Wong 2011), drawing order with g on top and y at the
# bottom, and legends sorted by wavelength
BAND_COLORS = {"u": "#56B4E9", "g": "#009E73", "r": "#D55E00", "i": "#E69F00", "z": "#CC79A7", "y": "#0072B2"}
BAND_ZORDER = {"u": 2, "g": 6, "r": 5, "i": 4, "z": 3, "y": 1}
BAND_ORDER = ["u", "g", "r", "i", "z", "y"]

fig, ax = plt.subplots(figsize=(9, 4))
entries = []                                   # (band, label, artist) for the sorted legend
for (sid, label, df, fx, fe, marker) in [
        (0, "ZTF det", ztf_det_flux, "flux", "flux_err", "o"),
        (0, "ZTF forced", ztf_fp_flux, "flux", "flux_err", "o"),
        (1, "LSST det", lsst_det, "psfflux", "psffluxerr", "s")]:
    for band, g in df.assign(sid=sid).pipe(decode).groupby("band_name"):
        artist = ax.errorbar(g.mjd, g[fx], g[fe], fmt=marker, ms=4, color=BAND_COLORS[band],
                             mfc="none" if "forced" in label else BAND_COLORS[band], zorder=BAND_ZORDER[band])
        entries.append((band, f"{label} {band}", artist))
for band, g in ztf_nd.assign(sid=0).pipe(decode).groupby("band_name"):
    ax.plot(g.mjd, mag_to_flux(g.diffmaglim), "v", color=BAND_COLORS[band], alpha=0.3, ms=4, zorder=0)
entries.sort(key=lambda e: BAND_ORDER.index(e[0]))   # stable sort: keeps ZTF det, ZTF forced, LSST within a band
ax.legend([e[2] for e in entries], [e[1] for e in entries], fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_xlim(61080, 61210)
ax.set_ylim(-0.05e6, 0.7e6)                  # a few shallow ZTF limits (bad nights) would stretch the axis
ax.set_xlabel("MJD")
ax.set_ylabel("difference flux [nJy]")
ax.set_title("SN 2026gzf: ZTF (circles) and LSST (squares); triangles = ZTF 5σ limits");

# %% [markdown]
# The two surveys agree. Small offsets are expected: ZTF and Rubin filters are not identical.
#
# ### 5.4 A reusable light-curve function
#
# The library `alerce_ms.get_lightcurve(oid, sid)` does all of the above for either survey and returns one
# standard table with `diff_flux`, `sci_flux` (both in nJy), `diff_mag`, `sci_mag`, `band_name`, `survey` and
# `kind` (`det` / `forced`). Read its source in `alerce_ms.py`: it is the same code as above.

# %% [markdown]
# ## 6. ZTF auxiliary tables
#
# ZTF alerts carry extra context, stored per detection (`measurement_id`):
#
# | Table | Content |
# |---|---|
# | `ztf_ps1` | 3 closest Pan-STARRS1 sources: magnitudes, star/galaxy score `sgscore` (1 = star), distance `distpsnr` |
# | `ztf_gaia` | nearest Gaia source and nearest bright Gaia source (distance, magnitude) |
# | `ztf_ss` | nearest known solar-system object (MPC): distance `ssdistnr`, magnitude, name (−999 = none) |
# | `ztf_dataquality` | image/PSF quality: FWHM, χ² of the PSF fit, number of bad pixels, zero points… |
# | `ztf_reference` | reference (template) image used for each band: dates, number of frames, nearest source in it |

# %%
tap_query(f"""
    SELECT TOP 3 measurement_id, objectidps1, sgmag1, srmag1, simag1, sgscore1, distpsnr1, nmtchps
    FROM alerce_tap.ztf_ps1 WHERE oid = {sn_oid}
""")

# %% [markdown]
# The closest PS1 source is 0.2″ away with `sgscore1 ≈ 0.03`: a **galaxy**. The supernova exploded near the
# centre of its host, as seen by Pan-STARRS.

# %%
tap_query(f"SELECT * FROM alerce_tap.ztf_gaia WHERE oid = {sn_oid}")

# %%
tap_query(f"SELECT TOP 3 measurement_id, ssdistnr, ssmagnr, ssnamenr FROM alerce_tap.ztf_ss WHERE oid = {sn_oid}")

# %%
tap_query(f"""
    SELECT TOP 5 measurement_id, fwhm, chipsf, classtar, nbad, seeratio, magzpsci
    FROM alerce_tap.ztf_dataquality WHERE oid = {sn_oid}
""")

# %%
ref = tap_query(f"""
    SELECT band, rfid, field, rcid, magnr, sigmagnr, mjdstartref, mjdendref, nframesref
    FROM alerce_tap.ztf_reference WHERE oid = {sn_oid}
""")
ref.assign(sid=0).pipe(decode)

# %% [markdown]
# The templates were built from 15–30 images taken in 2018–2021, long before the explosion: good.
#
# ## 7. LSST solar-system tables (`sid = 2`)
#
# Moving objects detected by Rubin are associated with known asteroids. `lsst_ss_detection` holds the observing
# geometry of every detection; `lsst_mpc_orbits` holds the Minor Planet Center orbit. `lsst_ss_object` (the
# per-asteroid summary, `sid = 2`) is not filled yet.

# %%
tap_query("SELECT TOP 3 * FROM alerce_tap.lsst_ss_object")

# %%
tap_query("""
    SELECT TOP 5 measurement_id, ssobjectid, designation, heliorange, toporange, phaseangle, elongation
    FROM alerce_tap.lsst_ss_detection
""")

# %%
orbits = tap_query("SELECT TOP 2000 ssobjectid, designation, a, e, i, q, h FROM alerce_tap.lsst_mpc_orbits")
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(orbits.a, orbits.e, s=3, c=orbits.i, cmap="viridis")
ax.set_xlim(0.5, 6)
ax.set_xlabel("semi-major axis a [au]")
ax.set_ylabel("eccentricity e")
ax.set_title("Orbits of asteroids detected by LSST (colour = inclination)");

# %% [markdown]
# Kirkwood gaps of the main belt are already visible near 2.5 and 2.8 au.
#
# ## 8. Derived products
#
# ### 8.1 `magstat`: per-band magnitude statistics
#
# One row per object and band: mean/median/min/max magnitude, first/last magnitude, `dmdt_first` (how fast it rose)…
# Currently filled for **ZTF only**.

# %%
tap_query(f"""
    SELECT oid, sid, band, n_det, magmean, magmin, magmax, magfirst, maglast, dmdt_first, saturation_rate
    FROM alerce_tap.magstat WHERE oid = {sn_oid} AND sid = 0
""").pipe(decode)

# %% [markdown]
# ### 8.2 `feature`: light-curve features in *long* format
#
# One row per `(oid, sid, feature_id, band)`. A few details:
# * the value column is called `feature_value` in TAP (`value` is a reserved word in ADQL);
# * `band = 0` means "no band" (e.g. `W1-W2`), and multi-digit bands are combinations (`12` = g+r);
# * only the **latest** version of each feature is kept (`version` → `feature_version_lut`).

# %%
rrl_oid = ztf_name_to_oid("ZTF17aabhbva")
feats = tap_query(f"""
    SELECT f.oid, f.sid, f.feature_id, f.band, f.version, f.feature_value
    FROM alerce_tap.feature AS f
    WHERE f.oid = {rrl_oid} AND f.sid = 0
""").pipe(decode)
print(len(feats), "feature values")
feats[feats.feature_name.isin(["Multiband_period", "Period_band", "Amplitude", "g-r_mean", "W1-W2", "PPE"])]

# %% [markdown]
# The RR Lyrae star has a period of **0.4889 days** in both g and r. We can pivot to a *wide* table with one column
# per feature and band:

# %%
feats["col"] = feats.feature_name + np.where(feats.band_name != "", "_" + feats.band_name, "")
feats.pivot_table(index="oid", columns="col", values="feature_value").iloc[:, :12]

# %% [markdown]
# It is usually better to filter features in ADQL by joining `feature_name_lut`, so the server only sends what you need:

# %%
tap_query(f"""
    SELECT f.oid, n.feature_name, f.band, f.feature_value
    FROM alerce_tap.feature AS f
    JOIN alerce_tap.feature_name_lut AS n ON n.sid = f.sid AND n.feature_id = f.feature_id
    WHERE f.oid = 313888627082919999 AND f.sid = 1
      AND n.feature_name IN ('Multiband_period', 'Period_band')
""").assign(sid=1).pipe(decode)

# %% [markdown]
# Interesting! This is the **LSST** counterpart of the same RR Lyrae star, and its multiband period is ~0.978 d,
# **twice** the ZTF value. Only the i band finds 0.489 d. We will come back to this alias in notebook 05.
#
# ### 8.3 `probability`: classifications
#
# One row per `(oid, sid, classifier_id, class_id)`, with the probability and its `ranking` within that classifier
# (1 = most probable class). `lastmjd` records which data the prediction used.

# %%
probs = tap_query(f"""
    SELECT oid, sid, classifier_id, class_id, classifier_version, probability, ranking, lastmjd
    FROM alerce_tap.probability
    WHERE oid = {sn_oid} AND sid = 0 AND ranking <= 3
""").pipe(decode)
probs.sort_values(["classifier_name", "ranking"])[["classifier_name", "class_name", "probability", "ranking"]]

# %% [markdown]
# The hierarchical light-curve classifier says **Transient** (top level, P = 0.95) and, within transients, **SESN**
# (stripped-envelope SN, P = 0.43). TNS classified it as a type Ic-BL: a stripped-envelope SN. 🎉
#
# > ⚠️ `classifier_version` in `probability` is a lossy integer ("2.2.0" → 220). Use `classifier.classifier_version`.
#
# **Selecting samples** is where `probability` shines. The partial index on `ranking = 1` makes this fast:
# the 10 most probable LSST supernova candidates according to the newest Rubin stamp classifier:

# %%
cid = classifier_lut.query("classifier_name == 'stamp_classifier_rubin_beta_20260421'").classifier_id.iloc[0]
sn_class = taxonomy_lut.query("classifier_id == @cid and class_name == 'SN'").class_id.iloc[0]
tap_query(f"""
    SELECT TOP 10 p.oid, p.sid, p.probability, o.meanra, o.meandec, o.firstmjd, o.n_det
    FROM alerce_tap.probability AS p
    JOIN alerce_tap.object AS o ON o.oid = p.oid AND o.sid = p.sid
    WHERE p.sid = 1 AND p.classifier_id = {cid} AND p.class_id = {sn_class}
      AND p.ranking = 1 AND p.probability > 0.8
    ORDER BY p.probability DESC
""")

# %% [markdown]
# ## 9. Crossmatches
#
# `xmatch` links each object to its nearest counterpart in external catalogs stored *inside* the database:
# AllWISE (`catid = 0`, mid-infrared + 2MASS) and Gaia DR3 (`catid = 1`). Join to the catalog tables on `oid_catalog`.
# Let us look at the LSST counterpart of the quasar:

# %%
tap_query("""
    SELECT x.catid, x.oid_catalog, x.dist,
           a.w1mpro, a.w2mpro, a.w3mpro, a.w4mpro, a.w1mpro - a.w2mpro AS w1_w2, a.k_m_2mass,
           g.phot_g_mean_mag, g.bp_rp, g.parallax, g.pm, g.in_qso_candidates, g.classprob_dsc_combmod_quasar
    FROM alerce_tap.xmatch AS x
    LEFT JOIN alerce_tap.allwise AS a ON x.catid = 0 AND a.oid_catalog = x.oid_catalog
    LEFT JOIN alerce_tap.gaiadr3_source AS g ON x.catid = 1 AND g.oid_catalog = x.oid_catalog
    WHERE x.oid = 313985345706262538 AND x.sid = 1
""").pipe(decode)

# %% [markdown]
# W1 − W2 = 1.3 (AGN have W1 − W2 > 0.8, Stern et al. 2012), zero parallax and proper motion, and Gaia's classifier
# gives P(quasar) = 0.998. A textbook quasar. In notebook 04 we add X-ray, UV and spectroscopic redshifts.
#
# ## 10. Rules of thumb
#
# 1. **Join on `(oid, sid)`**; ZTF extension tables have no `sid`, join them on `oid` (+ `measurement_id`).
# 2. **Decode integers with their key**: bands and features with `sid`, classes with `classifier_id`.
# 3. **Use `TOP N`** while exploring, and filter on indexed columns (`oid`, `sid`, `ranking = 1`, `probability`,
#    positions via `CONTAINS`).
# 4. **Select the page first, then join** large tables (sub-query with `TOP` → join `feature`, `detection`, …).
# 5. Avoid `COUNT(*)` or filters on non-indexed columns over the huge tables (`detection` has ~660 million rows,
#    `feature` ~1.4 billion): they can take minutes.
# 6. ZTF: magnitudes, UTC, error 100 = n/a. LSST: nJy fluxes, TAI, `m = 31.4 − 2.5 log10 F`.
# 7. There is **no** LSST non-detection table and **no** LSST `magstat` yet.
#
# ### The `alerce_ms` library
#
# Everything we wrote here (plus caching, retries, the REST API, catsHTM and the experimental explorer) is collected
# in `alerce_ms.py`. From now on we just import it:

# %%
import alerce_ms as am

[name for name in dir(am) if not name.startswith("_") and callable(getattr(am, name))]

# %%
am.get_probabilities("ZTF20acyxpwi", sid=0).query("ranking == 1")[["classifier_name", "class_name", "probability"]]

# %% [markdown]
# ### ✏️ Exercises
#
# 1. **(easy)** Which bands has the LSST counterpart of the supernova been detected in? Use `am.get_lightcurve` and
#    `groupby("band_name")`.
# 2. **(easy)** Decode `feature.band = 12` for ZTF and for LSST with `am.band_label`. Do they mean the same thing?
# 3. **(medium)** Write an ADQL query returning the 20 ZTF objects with the highest probability of being `RRLab`
#    according to `lc_classifier_BHRF_forced_phot_periodic`, together with their `Multiband_period`.
#    Hint: select the page in a sub-query, then join `feature` and `feature_name_lut`.
# 4. **(medium)** For the quasar `ZTF20acyxpwi`, find the `ztf_ps1` star/galaxy score and the `ztf_gaia` nearest
#    source. Are they consistent with what Gaia DR3 says in `xmatch`?
# 5. **(challenge)** Plot the distribution of `h` (absolute magnitude) of LSST asteroids in `lsst_mpc_orbits` for
#    `a < 2.5 au` and `a > 2.8 au`. Which population is fainter on average? Why?

# %%
# your code here

# %% [markdown]
# **Next:** [02 · One source, two surveys](02_one_source_two_surveys.ipynb)

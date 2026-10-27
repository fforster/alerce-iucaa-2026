# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/02_one_source_two_surveys.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 02 · One source, two surveys
#
# **ALeRCE multisurvey tutorial — IUCAA 2026** · ~25 min
#
# We follow three sources in the COSMOS field that were observed by **both ZTF and LSST**:
#
# | ZTF name | What it is |
# |---|---|
# | `ZTF26aaonmha` | SN 2026gzf, a type Ic-BL supernova |
# | `ZTF20acyxpwi` | a quasar |
# | `ZTF17aabhbva` | an RR Lyrae star |
#
# You will learn to
# 1. find the LSST counterpart(s) of a ZTF object (and vice versa);
# 2. use the **multisurvey REST API** endpoint by endpoint;
# 3. compare light curves, classifications and image stamps from the two surveys;
# 4. use the **experimental explorer** for crossmatches and spectroscopic redshifts.

# %%
import sys

if "google.colab" in sys.modules:
    !pip -q install pyvo astroquery
    !wget -q -N https://raw.githubusercontent.com/fforster/alerce-iucaa-2026/main/alerce_ms.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alerce_ms as am

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

# %% [markdown]
# <img src="https://raw.githubusercontent.com/fforster/alerce-iucaa-2026/main/figures/database_schematic.png" width="480" alt="What ALeRCE stores about one object"/>
#
# *Reminder: what ALeRCE stores about each object (see notebook 01). Here we fill every box for the same source twice,
# once for ZTF and once for LSST.*

# %% [markdown]
# ## 1. Finding counterparts across surveys
#
# Remember **rule 2** from notebook 01: there is no cross-survey object association. The link between a ZTF object
# and an LSST object is purely **positional**. The light-curve API has a cone search that looks in *all* surveys at
# once (`lightcurve_api/conesearch/objects_by_coordinates`; its radius is in degrees). `am.find_counterparts` wraps it.
#
# First we need the position of each ZTF object, from `object_api/object`:

# %%
names = {"ZTF26aaonmha": "SN Ic-BL", "ZTF20acyxpwi": "quasar", "ZTF17aabhbva": "RR Lyrae"}

ztf_objects = pd.DataFrame([am.get_object(name, "ztf") for name in names])
ztf_objects.insert(0, "name", list(names))
ztf_objects.insert(1, "type", list(names.values()))
ztf_objects[["name", "type", "oid", "meanra", "meandec", "firstmjd", "lastmjd", "n_det", "n_forced", "n_non_det"]]

# %%
matches = []
for row in ztf_objects.itertuples():
    c = am.find_counterparts(row.meanra, row.meandec, radius_arcsec=2.0)
    c.insert(0, "ztf_name", row.name)
    matches.append(c)
matches = pd.concat(matches, ignore_index=True)
matches[["ztf_name", "name", "survey", "sid", "oid", "sep_arcsec"]]

# %% [markdown]
# * The supernova and the quasar each have **one** LSST counterpart within 0.5″.
# * The RR Lyrae star has **two** LSST diaObjects. Let us look at them:

# %%
rrl_lsst = matches.query("ztf_name == 'ZTF17aabhbva' and survey == 'lsst'").oid.tolist()
pd.DataFrame([am.get_object(oid, "lsst") for oid in rrl_lsst])[
    ["oid", "meanra", "meandec", "firstmjd", "lastmjd", "n_det", "n_forced"]]

# %% [markdown]
# Both objects are at the same position (≈0.04″ from the ZTF position) and their time ranges overlap. Rubin's
# alert-production association sometimes splits one star into several diaObjects (for example after
# reprocessing, or when sources are de-blended differently). **When you combine LSST data for a star, check for
# duplicate diaObjects and merge them.** We do exactly that in notebook 05.
#
# The same cone search can be done with ADQL on the `object` table (`am.tap_cone`), useful when you want extra
# columns or many positions:

# %%
sn = ztf_objects.iloc[0]
am.tap_cone(sn.meanra, sn.meandec, radius_arcsec=2.0).pipe(am.decode)

# %% [markdown]
# Let us store the pairs we will use:

# %%
pairs = {
    "ZTF26aaonmha": [314003014107006318],
    "ZTF20acyxpwi": [313985345706262538],
    "ZTF17aabhbva": rrl_lsst,
}

# %% [markdown]
# ## 2. The multisurvey REST API, endpoint by endpoint
#
# Each service lives at `https://api-lsst.alerce.online/<name>_api/<endpoint>` and documents itself at
# `.../<name>_api/docs`. `am.api_get(service, endpoint, **params)` is a three-line wrapper around `requests.get`.
#
# ### 2.1 `object_api`
#
# Besides `/object` (used above), `/list_objects` filters by classifier, class, probability, dates and position —
# the same query that powers the explorers. Here: LSST objects classified as AGN within 1° of COSMOS.
# (Radius in arcsec here. Note: the `total` field of the answer is not reliable at the moment; use paging.)

# %%
res = am.api_get("object", "list_objects", survey="lsst", classifier="stamp_classifier_rubin_beta_20260421",
                 class_name="AGN", probability=0.5, ra=150.1, dec=2.2, radius=3600, page_size=5)
pd.DataFrame(res["items"])[["oid", "meanra", "meandec", "n_det", "class_name", "probability"]]

# %% [markdown]
# ### 2.2 `lightcurve_api`
#
# `/lightcurve` returns detections, non-detections and forced photometry in one call. Each detection carries the full
# native record of its survey: for LSST, Rubin's DiaSource columns (`psfFlux`, `scienceFlux`, `reliability`, …).

# %%
lc_json = am.api_get("lightcurve", "lightcurve", oid=314003014107006318, survey_id="lsst")
print({k: len(v) for k, v in lc_json.items()})
pd.DataFrame(lc_json["detections"]).iloc[:3, :14]

# %% [markdown]
# For ZTF use the **integer** `oid` with `/detections` and `/non_detections`. (At the time of writing, the ZTF
# `/lightcurve` and `/forced-photometry` endpoints fail; get ZTF forced photometry from TAP with
# `am.get_lightcurve`.)

# %%
sn_oid = am.ztf_name_to_oid("ZTF26aaonmha")
ztf_det = pd.DataFrame(am.api_get("lightcurve", "detections", oid=sn_oid, survey_id="ztf"))
ztf_det[["mjd", "band", "magpsf", "sigmapsf", "isdiffpos", "rb", "drb", "has_stamp"]].head()

# %% [markdown]
# ### 2.3 `probability_api`, `magstat_api`, `crossmatch_api`, `classifier_api`

# %%
pd.DataFrame(am.api_get("probability", "probability", oid=314003014107006318)).sort_values(
    ["classifier_name", "ranking"]).head(10)

# %% [markdown]
# ⚠️ The API is not yet consistent about ZTF identifiers: the `lightcurve_api` and `probability_api` endpoints want
# the integer `oid`, while `magstat_api` and `crossmatch_api` want the ZTF name (`object_api` accepts both).
# If one form fails (HTTP 400/500), try the other.

# %%
pd.DataFrame(am.api_get("magstat", "magstats", oid="ZTF26aaonmha", survey_id="ztf"))[
    ["ndet", "magmean", "magmin", "magmax", "magfirst", "maglast", "firstmjd", "lastmjd"]]

# %% [markdown]
# `crossmatch_api` returns the nearest source in each of the catalogs of the ALeRCE **catsHTM** service (2MASS, WISE,
# GALEX, XMM, SDSS spectra, NED redshifts, Gaia, …). We use it heavily for the quasar in notebook 04:

# %%
xm = am.api_get("crossmatch", "crossmatch", oid="ZTF20acyxpwi", survey_id="ztf")
{list(item)[0]: round(list(item.values())[0]["distance"]["value"], 2) for item in xm}   # catalog: distance ["]

# %% [markdown]
# ### 2.4 API or TAP?
#
# | | API | TAP |
# |---|---|---|
# | one object, everything about it | ✅ simple JSON | ✅ several queries |
# | many objects / samples | one request per object 😬 | ✅ one query |
# | arbitrary joins, features, catalog columns | ❌ | ✅ |
# | ZTF forced photometry, features | ❌ (for now) | ✅ |
#
# `am.get_lightcurve(oid, sid)` uses TAP and returns the same columns for both surveys. Let us use it from now on.
# To see what else TAP offers, browse the [alerce_tap schema explorer](https://claude.ai/artifact/JEpEyijsTpD1y83ut7Zia7).

# %% [markdown]
# ## 3. Light curves from both surveys
#
# For each pair we download the ZTF and LSST light curves (detections + forced photometry):

# %%
lcs = {}
for ztf_name, lsst_oids in pairs.items():
    parts = [am.get_lightcurve(ztf_name, 0)] + [am.get_lightcurve(oid, 1) for oid in lsst_oids]
    lcs[ztf_name] = pd.concat(parts, ignore_index=True)
    print(ztf_name, lcs[ztf_name].groupby("survey").size().to_dict())

# %% [markdown]
# Which quantity should we plot?
# * **Supernova:** the *difference* flux — the host galaxy is removed by the subtraction.
# * **Quasar and RR Lyrae:** the *total* brightness (`sci_mag`): the template already contains the source at some
#   average brightness, so the difference flux would oscillate around zero.

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 11))
am.plot_lightcurve(lcs["ZTF26aaonmha"], flux="diff", ax=axes[0], title="SN 2026gzf (difference flux)")
axes[0].set_xlim(61080, 61210)
am.plot_lightcurve(lcs["ZTF20acyxpwi"], flux="sci", mag=True, ax=axes[1], title="Quasar (total magnitude)",
                   show_forced=False)
rrl_ztf = lcs["ZTF17aabhbva"].query("survey == 'ztf'")
am.plot_lightcurve(rrl_ztf, flux="sci", mag=True, ax=axes[2],
                   title="RR Lyrae, ZTF only (filled: alert detections, open: forced photometry)")
plt.tight_layout()

# %% [markdown]
# Look at the RR Lyrae panel: the ZTF **alert detections** (filled) pile up at two levels, ~0.5 mag above and below
# the mean, with a gap in between, while the **forced photometry** (open; available for this star since ~MJD 60250)
# also lands inside the gap. Why? An alert is only
# issued when the source differs from the *template* by ≥ 5σ. The template shows the star at its average brightness,
# so epochs near the average never produce alerts. **Alert light curves of variable stars are biased; use forced
# photometry when you can.**

# %% [markdown]
# Things to notice:
# * ZTF has a **long baseline** (since 2018) but only g, r, i; LSST adds u, z, y, many more points per night
#   (COSMOS is a deep drilling field) and much better photometric precision.
# * The quasar shows slow, stochastic variability over years, the RR Lyrae star a large-amplitude scatter that will
#   become a clean curve once we fold it with the right period (notebook 05).
# * ZTF `magpsf_corr`/`mag_corr` (apparent magnitudes) and LSST `scienceFlux` are measured differently (reference
#   flux added back vs. flux on the direct image), and the filters differ: small offsets between surveys are normal.

# %% [markdown]
# ## 4. Classifications in the two surveys
#
# ZTF objects are classified by the light-curve classifier (full history) and the ZTF stamp classifier (first alert);
# LSST objects, for now, by the Rubin stamp classifiers. Let us compare the top class of each:

# %%
rows = []
for ztf_name, lsst_oids in pairs.items():
    for sid, oid in [(0, ztf_name)] + [(1, o) for o in lsst_oids]:
        p = am.get_probabilities(oid, sid).query("ranking == 1")
        for r in p.itertuples():
            rows.append(dict(source=names[ztf_name], survey=am.SURVEY[sid], oid=am.as_oid(oid),
                             classifier=r.classifier_name, top_class=r.class_name,
                             probability=round(r.probability, 3)))
pd.DataFrame(rows)

# %% [markdown]
# * The ZTF light-curve classifier gets all three right: **Transient → SESN**, **Stochastic → QSO**, **Periodic → RRLab**.
# * The stamp classifiers only see the image triplet of the **first** alert. For the supernova, ZTF's first alert was
#   called *bogus*, and so was LSST's — whose first detection was a weak residual on the host galaxy, weeks before
#   the explosion (see notebook 01). For the quasar, the ZTF stamp classifier says AGN, while the Rubin (beta)
#   stamp classifiers say VS (variable star): from a single stamp a point-like variable quasar looks like a star.
# * Lessons: **a classification depends on the data the model saw** (`lastmjd`); stamp classes are a fast first
#   guess, light-curve classes are more reliable; and combining surveys gives you more information than either alone.

# %% [markdown]
# ## 5. Image stamps
#
# Every alert carries three small images: the **science** image, the **template** (reference) and their
# **difference**. Let us compare ZTF and LSST stamps of the supernova near maximum light.

# %%
lc = lcs["ZTF26aaonmha"]
ztf_peak = lc.query("survey == 'ztf' and kind == 'det'").sort_values("diff_flux").iloc[-1]
lsst_dets = pd.DataFrame(lc_json["detections"]).query("has_stamp")
lsst_peak = lsst_dets.sort_values("psfFlux").iloc[-1]
print(f"ZTF  peak detection: MJD {ztf_peak.mjd:.2f}, band {ztf_peak.band_name}")
print(f"LSST peak detection: MJD {lsst_peak.mjd:.2f}, band {lsst_peak.band_map[str(lsst_peak.band)]}")

# where on the sky is the LSST cut-out? (its 4 corners, from its WCS) and where is the ZTF cut-out centred?
lsst_corners = am.stamp_corners(314003014107006318, 1, int(lsst_peak.measurement_id))
ztf_center = am.stamp_center(sn_oid, 0, int(ztf_peak.measurement_id))
footprint = dict(corners=lsst_corners, center=ztf_center, scale_arcsec=am.ZTF_PIXEL_ARCSEC, label="LSST stamp")

fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
am.plot_stamps(am.get_stamps(sn_oid, 0, int(ztf_peak.measurement_id)), "ZTF", axes[0], footprint=footprint)
am.plot_stamps(am.get_stamps(314003014107006318, 1, int(lsst_peak.measurement_id)), "LSST", axes[1])
plt.tight_layout()

# %% [markdown]
# ZTF stamps are 63×63 pixels of 1.0″ (≈1′ across); LSST stamps are 51×51 pixels of 0.2″ (≈10″ across). The blue
# square on the ZTF stamps is the **footprint of the LSST stamp**: the whole LSST cut-out (bottom row) covers only that
# small, tilted patch of the ZTF one. It is tilted because Rubin's camera was rotated with respect to North when this
# image was taken. With LSST we see the structure of the host galaxy and the supernova on top of it.
#
# **Orientation.** Both rows are shown with **North up and East left** (arrows on the left panels), as on the sky.
# Raw LSST cut-outs come in the orientation of the detector, rotated by an arbitrary angle (here ≈ 51°). Their FITS
# header carries a **WCS** (World Coordinate System): a matrix that converts a step in pixels into a step on the sky.
# `am.north_up` inverts that matrix to find which pixel direction points North and which points East, and resamples
# the image so North is up — the same idea the experimental explorer uses. ZTF cut-outs have no WCS but are already
# aligned with the sky, so we only flip their rows. Try `am.get_stamps(..., north=False)` to see the raw orientation.

# %% [markdown]
# ## 6. The experimental explorer
#
# The experimental explorer (ALeRCE Labs) shows ZTF and LSST data together, plus ZTF data-release light curves,
# crossmatches with many catalogs and spectroscopic redshifts. It also exposes a small JSON API. `am.explorer_xmatch`
# returns what the explorer overlays on the sky around an object:

# %%
for ztf_name in pairs:
    am.show_link(ztf_name, "ztf", f"{ztf_name} ({names[ztf_name]}) in the experimental explorer")

# %%
am.explorer_xmatch("ZTF26aaonmha", "ztf").head(10)

# %% [markdown]
# For the supernova the explorer finds the TNS name (SN 2026gzf, type Ic-BL in VSX/TNS) and, in the `host`
# category, its host galaxy with spectroscopic redshift z ≈ 0.0345 from NED, 2dFGRS, HECATE, GLADE and DESI.
# Notebook 03 uses this, together with the DELIGHT host finder.

# %%
for ztf_name in ["ZTF20acyxpwi", "ZTF17aabhbva"]:
    x = am.explorer_xmatch(ztf_name, "ztf")
    print(ztf_name, names[ztf_name])
    display(x[x.category != "host"].head(5)[["cat_name", "category", "name", "type", "z", "sep"]])

# %% [markdown]
# ### ✏️ Exercises
#
# 1. **(easy)** Start from the LSST side: take the LSST quasar counterpart `313985345706262538`, get its position with
#    `am.get_object(..., "lsst")` and recover its ZTF counterpart with `am.find_counterparts`.
# 2. **(easy)** How many LSST *detections* has each of the two RR Lyrae diaObjects? Do they share any `measurement_id`?
# 3. **(medium)** Compare the LSST `reliability` of the supernova detections before MJD 61110 and after. What do you
#    conclude about the early "detections"?
# 4. **(medium)** Use `/list_objects` with `survey="ztf"` and the light-curve classifier to find ZTF RR Lyrae
#    (`class_name="RRLab"`) within 1° of COSMOS (ra = 150.1, dec = 2.2) and look for their LSST counterparts.
#    (Keep it small — at most ~10 cone searches.)
# 5. **(challenge)** Plot the difference stamps of the supernova from LSST at five epochs along the light curve.

# %%
# your code here

# %% [markdown]
# **Next:** [03 · Supernovae](03_supernovae.ipynb)

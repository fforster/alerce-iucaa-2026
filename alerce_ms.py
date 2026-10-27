"""
alerce_ms.py — small helper library for the ALeRCE multisurvey tutorial (IUCAA 2026).

Everything here is a thin wrapper around three public services:

* TAP    https://tap.alerce.online/tap           (schema ``alerce_tap``, ADQL queries)
* API    https://api-lsst.alerce.online/<name>_api (multisurvey REST API, JSON)
* Labs   https://explorer-experiment.alerce.online (experimental explorer)

plus the ALeRCE catsHTM crossmatch service (https://catshtm.alerce.online).

The functions are deliberately short so that you can read them: open this file
next to the notebooks and see exactly which query is sent to which service.

Conventions used throughout
---------------------------
* An object is identified by the pair ``(oid, sid)``.  ``sid`` = 0 ZTF,
  1 LSST DIA object, 2 LSST solar-system object.  Always join on both columns.
* ZTF ``oid`` in the multisurvey database is an integer that encodes the ZTF
  name (``ztf_name_to_oid``).  LSST ``oid`` is the Rubin ``diaObjectId``.
* Fluxes are in nJy.  AB magnitude = 31.4 - 2.5 log10(flux / nJy).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests

TAP_URL = "https://tap.alerce.online/tap"
API_URL = "https://api-lsst.alerce.online"
EXPLORER_URL = "https://explorer-experiment.alerce.online"
CATSHTM_URL = "https://catshtm.alerce.online"

SID = {"ztf": 0, "lsst": 1, "lsst_ss": 2}
SURVEY = {0: "ztf", 1: "lsst", 2: "lsst"}

# colour-blind friendly colours per band name (ZTF and LSST share g, r, i)
BAND_COLORS = {"u": "#1f3ab4", "g": "#56b4e9", "r": "#d55e00", "i": "#cc79a7",
               "z": "#009e73", "y": "#7f6a00"}
SURVEY_MARKERS = {"ztf": "o", "lsst": "s"}

AB_ZP_NJY = 31.4  # AB zero point for fluxes in nJy

# ---------------------------------------------------------------------------
# 0. Being a good citizen: on-disk cache + retries
# ---------------------------------------------------------------------------
# The ALeRCE services are shared by everybody and rate-limited per IP address
# (HTTP 429 "Too Many Requests").  In a classroom all of you share one IP, so:
#   * every query result is cached on disk: re-running a cell costs nothing;
#   * if the server says 429 we wait and retry instead of failing.
CACHE_DIR = os.environ.get("ALERCE_CACHE", os.path.expanduser("~/.cache/alerce_ms"))
USE_CACHE = True


def _cache_path(key: str, ext: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ext)


def _with_retries(func, *args, retries: int = 6, **kwargs):
    """Call func; on HTTP 429 / 5xx wait 5, 10, 20, ... seconds and try again."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # requests or pyvo errors
            msg = str(e)
            transient = "429" in msg or "Too Many" in msg or "502" in msg or "503" in msg \
                or "504" in msg or "timed out" in msg.lower()
            if not transient or attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  server busy ({msg[:60]}...), retrying in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# 1. TAP
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def tap_service():
    import pyvo
    return pyvo.dal.TAPService(TAP_URL)


def tap_query(adql: str, maxrec: int = 1_000_000, cache: bool = True) -> pd.DataFrame:
    """Run a synchronous ADQL query against the ALeRCE TAP service.

    Results are cached on disk (set ``cache=False`` to force a new query)."""
    path = _cache_path(adql + str(maxrec), ".pkl")
    if cache and USE_CACHE and os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:  # cache written by another pandas/numpy version
            pass
    table = _with_retries(lambda: tap_service().search(adql, maxrec=maxrec).to_table())
    df = table.to_pandas()
    df.to_pickle(path)
    return df


def tap_tables() -> pd.DataFrame:
    """List the tables published in the multisurvey TAP schema."""
    return tap_query("SELECT table_name, description FROM TAP_SCHEMA.tables "
                     "WHERE schema_name = 'alerce_tap' ORDER BY table_name")


def tap_columns(table: str) -> pd.DataFrame:
    """List the columns of one table (name without the ``alerce_tap.`` prefix)."""
    return tap_query("SELECT column_name, datatype, unit, description "
                     "FROM TAP_SCHEMA.columns "
                     f"WHERE table_name = 'alerce_tap.{table}'")


# ---------------------------------------------------------------------------
# 2. Identifiers
# ---------------------------------------------------------------------------
def ztf_name_to_oid(name: str) -> int:
    """'ZTF26aaonmha' -> 36029005852662990 (integer oid used in alerce_tap).

    The ZTF name is a 2-digit year plus 7 letters, i.e. a base-26 number.
    ALeRCE adds a survey prefix (ZTF = 1) in the top 8 bits: 1 << 55.
    """
    year, letters = int(name[3:5]), name[5:12]
    seq = sum((ord(c) - ord("a")) * 26 ** (6 - i) for i, c in enumerate(letters))
    return (1 << 55) + year * 26 ** 7 + seq


def oid_to_ztf_name(oid: int) -> str:
    """36029005852662990 -> 'ZTF26aaonmha' (inverse of ztf_name_to_oid)."""
    value = int(oid) & ((1 << 55) - 1)
    year, seq = (value // 26 ** 7) % 100, value % 26 ** 7
    letters = ""
    for i in range(6, -1, -1):
        letters += chr(seq // 26 ** i + ord("a"))
        seq %= 26 ** i
    return f"ZTF{year:02d}{letters}"


def as_oid(oid) -> int:
    """Accept a ZTF name or an integer-like oid and return the integer oid."""
    if isinstance(oid, str) and oid.startswith("ZTF"):
        return ztf_name_to_oid(oid)
    return int(oid)


def display_name(oid, sid: int) -> str:
    """Human-friendly name: the ZTF name for ZTF, the diaObjectId for LSST."""
    return oid_to_ztf_name(oid) if sid == 0 else str(oid)


# ---------------------------------------------------------------------------
# 3. Look-up tables
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_luts() -> dict:
    """Download all lookup tables once and keep them in memory."""
    names = {
        "sid": "SELECT sid, tid, survey_name FROM alerce_tap.sid_lut",
        "band": "SELECT sid, band, band_name, band_order FROM alerce_tap.band",
        "classifier": "SELECT classifier_id, classifier_name, classifier_version, tid "
                      "FROM alerce_tap.classifier",
        "taxonomy": "SELECT classifier_id, class_id, class_name, taxonomy_order "
                    "FROM alerce_tap.taxonomy",
        "feature_name": "SELECT sid, feature_id, feature_name FROM alerce_tap.feature_name_lut",
        "feature_version": "SELECT sid, version_id, version_name "
                           "FROM alerce_tap.feature_version_lut",
        "catalog": "SELECT catid, catalog_name FROM alerce_tap.catalog_id_lut",
    }
    return {key: tap_query(q) for key, q in names.items()}


def classifier_id(name: str) -> int:
    luts = load_luts()["classifier"]
    return int(luts.loc[luts.classifier_name == name, "classifier_id"].iloc[0])


def class_id(classifier_name: str, class_name: str) -> int:
    tax = load_luts()["taxonomy"]
    row = tax[(tax.classifier_id == classifier_id(classifier_name)) & (tax.class_name == class_name)]
    return int(row.class_id.iloc[0])


def feature_ids(sid: int, names) -> dict:
    """{feature_name: feature_id} for one survey (ids differ between surveys!)."""
    lut = load_luts()["feature_name"]
    lut = lut[(lut.sid == sid) & lut.feature_name.isin(list(names))]
    return dict(zip(lut.feature_name, lut.feature_id))


def band_label(sid: int, band: int) -> str:
    """Decode a band number, including feature band combinations (12 -> 'gr', 0 -> '')."""
    lut = load_luts()["band"]
    names = dict(zip(lut[lut.sid == sid].band, lut[lut.sid == sid].band_name))
    if band in names:
        return names[band]
    if band == 0:
        return ""
    return "".join(names.get(int(d), "?") for d in str(band))


def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable columns for every integer code found in ``df``.

    sid -> survey_name, (sid, band) -> band_name, classifier_id -> classifier_name,
    (classifier_id, class_id) -> class_name, (sid, feature_id) -> feature_name,
    (sid, version) -> version_name, catid -> catalog_name.
    """
    luts = load_luts()
    out = df.copy()
    if "sid" in out:
        out = out.merge(luts["sid"][["sid", "survey_name"]], on="sid", how="left")
        if "band" in out:
            out["band_name"] = [band_label(s, b) for s, b in zip(out.sid, out.band)]
        if "feature_id" in out:
            out = out.merge(luts["feature_name"], on=["sid", "feature_id"], how="left")
        if "version" in out:
            v = luts["feature_version"].rename(columns={"version_id": "version"})
            out = out.merge(v, on=["sid", "version"], how="left")
    if "classifier_id" in out:
        out = out.merge(luts["classifier"][["classifier_id", "classifier_name"]],
                        on="classifier_id", how="left")
        if "class_id" in out:
            out = out.merge(luts["taxonomy"][["classifier_id", "class_id", "class_name"]],
                            on=["classifier_id", "class_id"], how="left")
    if "catid" in out:
        out = out.merge(luts["catalog"], on="catid", how="left")
    return out


# ---------------------------------------------------------------------------
# 4. Multisurvey REST API and cross-survey matching
# ---------------------------------------------------------------------------
def get_json(url: str, params: dict | None = None, timeout: float = 60, cache: bool = True):
    """HTTP GET returning JSON, with on-disk cache and retries on 429/5xx."""
    params = params or {}
    path = _cache_path(url + json.dumps(params, sort_keys=True, default=str), ".json")
    if cache and USE_CACHE and os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    def _get():
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    data = _with_retries(_get)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def api_get(service: str, endpoint: str, timeout: float = 60, **params):
    """GET https://api-lsst.alerce.online/<service>_api/<endpoint>?params -> JSON."""
    return get_json(f"{API_URL}/{service}_api/{endpoint}", params, timeout)


def get_object(oid, survey: str) -> dict:
    """Basic object information from the multisurvey API."""
    return api_get("object", "object", oid=oid, survey_id=survey)


def find_counterparts(ra: float, dec: float, radius_arcsec: float = 2.0,
                      neighbors: int = 10) -> pd.DataFrame:
    """All ZTF *and* LSST objects within ``radius_arcsec`` of (ra, dec).

    There is no cross-survey object association in the database: the same star
    is one object in ZTF and another in LSST.  A positional match is how you link
    them.  (The API's ``radius`` is in degrees.)
    """
    rows = api_get("lightcurve", "conesearch/objects_by_coordinates", ra=ra, dec=dec,
                   radius=radius_arcsec / 3600, neighbors=neighbors)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"objectId": "name", "survey_id": "survey"})
    df["sid"] = df.survey.map(SID)
    df["oid"] = [as_oid(n) for n in df.name]
    df["sep_arcsec"] = angular_sep_arcsec(ra, dec, df.ra, df.dec)
    return df.sort_values("sep_arcsec").reset_index(drop=True)


def tap_cone(ra: float, dec: float, radius_arcsec: float, sid: int | None = None) -> pd.DataFrame:
    """Same as ``find_counterparts`` but with ADQL on the ``object`` table."""
    where_sid = "" if sid is None else f"AND sid = {sid}"
    return tap_query(f"""
        SELECT oid, sid, meanra, meandec, firstmjd, lastmjd, n_det, n_forced,
               3600 * DISTANCE(POINT('ICRS', meanra, meandec), POINT('ICRS', {ra}, {dec})) AS sep_arcsec
        FROM alerce_tap.object
        WHERE 1 = CONTAINS(POINT('ICRS', meanra, meandec), CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600}))
        {where_sid}
        ORDER BY sep_arcsec""")


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, np.asarray(ra2, float), np.asarray(dec2, float)))
    s = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(s))) * 3600


# ---------------------------------------------------------------------------
# 5. Light curves (TAP), returned in one common format for ZTF and LSST
# ---------------------------------------------------------------------------
def mag_to_flux(mag):
    return 10 ** (-0.4 * (np.asarray(mag, float) - AB_ZP_NJY))


def flux_to_mag(flux):
    with np.errstate(invalid="ignore", divide="ignore"):
        return AB_ZP_NJY - 2.5 * np.log10(np.asarray(flux, float))


def _ztf_lightcurve(oid: int, forced: bool) -> pd.DataFrame:
    if forced:
        q = f"""
        SELECT f.mjd, f.band, f.measurement_id, x.mag, x.e_mag, x.isdiffpos,
               x.mag_corr, x.e_mag_corr
        FROM alerce_tap.forced_photometry AS f
        JOIN alerce_tap.ztf_forced_photometry AS x
          ON x.oid = f.oid AND x.measurement_id = f.measurement_id
        WHERE f.oid = {oid} AND f.sid = 0"""
    else:
        q = f"""
        SELECT d.mjd, d.band, d.measurement_id, x.magpsf AS mag, x.sigmapsf AS e_mag,
               x.isdiffpos, x.magpsf_corr AS mag_corr, x.sigmapsf_corr AS e_mag_corr
        FROM alerce_tap.detection AS d
        JOIN alerce_tap.ztf_detection AS x
          ON x.oid = d.oid AND x.measurement_id = d.measurement_id
        WHERE d.oid = {oid} AND d.sid = 0"""
    df = tap_query(q)
    # 100 is the "not available" code for ZTF errors
    for col in ("e_mag", "e_mag_corr"):
        df.loc[df[col] >= 100, col] = np.nan
    sign = np.where(df.isdiffpos < 0, -1.0, 1.0)
    df["diff_flux"] = sign * mag_to_flux(df.mag)
    df["diff_flux_err"] = np.abs(df.diff_flux) * 0.4 * np.log(10) * df.e_mag
    df["sci_flux"] = mag_to_flux(df.mag_corr)
    df["sci_flux_err"] = df.sci_flux * 0.4 * np.log(10) * df.e_mag_corr
    return df[["mjd", "band", "measurement_id", "diff_flux", "diff_flux_err",
               "sci_flux", "sci_flux_err"]]


def _lsst_lightcurve(oid: int, forced: bool) -> pd.DataFrame:
    common, ext = ("forced_photometry", "lsst_forced_photometry") if forced \
        else ("detection", "lsst_detection")
    q = f"""
    SELECT c.mjd, c.band, c.measurement_id,
           x.psfflux AS diff_flux, x.psffluxerr AS diff_flux_err,
           x.scienceflux AS sci_flux, x.sciencefluxerr AS sci_flux_err
    FROM alerce_tap.{common} AS c
    JOIN alerce_tap.{ext} AS x
      ON x.oid = c.oid AND x.measurement_id = c.measurement_id AND x.sid = c.sid
    WHERE c.oid = {oid} AND c.sid = 1"""
    return tap_query(q)


def get_lightcurve(oid, sid: int, forced: bool = True, detections: bool = True) -> pd.DataFrame:
    """Light curve of one object in a survey-independent format.

    Columns: mjd, band_name, survey, kind ('det' or 'forced'), diff_flux(_err)
    [difference-image flux, nJy] and sci_flux(_err) [total flux at the position,
    nJy], plus the corresponding AB magnitudes diff_mag / sci_mag.

    * ZTF: built from magpsf/mag (difference) and magpsf_corr/mag_corr
      (apparent magnitude, only when ALeRCE could correct for the reference flux).
    * LSST: psfflux (difference) and scienceflux (direct image), already in nJy.
    """
    oid = as_oid(oid)
    fetch = _ztf_lightcurve if sid == 0 else _lsst_lightcurve
    parts = []
    if detections:
        parts.append(fetch(oid, forced=False).assign(kind="det"))
    if forced:
        parts.append(fetch(oid, forced=True).assign(kind="forced"))
    lc = pd.concat(parts, ignore_index=True)
    lc["survey"] = SURVEY[sid]
    lc["oid"] = oid
    lc["band_name"] = [band_label(sid, b) for b in lc.band]
    lc["diff_mag"] = flux_to_mag(lc.diff_flux)
    lc["diff_mag_err"] = 1.0857 * lc.diff_flux_err / lc.diff_flux.abs()
    lc["sci_mag"] = flux_to_mag(lc.sci_flux)
    lc["sci_mag_err"] = 1.0857 * lc.sci_flux_err / lc.sci_flux
    return lc.sort_values("mjd").reset_index(drop=True)


def get_ztf_non_detections(oid) -> pd.DataFrame:
    """5-sigma upper limits (ZTF only — LSST has no non-detection table)."""
    df = tap_query(f"SELECT mjd, band, diffmaglim FROM alerce_tap.ztf_non_detection "
                   f"WHERE oid = {as_oid(oid)} ORDER BY mjd")
    df["band_name"] = [band_label(0, b) for b in df.band]
    return df


# ---------------------------------------------------------------------------
# 6. Features, probabilities, magstats, crossmatches (TAP)
# ---------------------------------------------------------------------------
def get_features(oid, sid: int, names=None, wide: bool = False) -> pd.DataFrame:
    """Features of one object, decoded.  ``names`` restricts to some feature names."""
    oid = as_oid(oid)
    where = ""
    if names is not None:
        ids = feature_ids(sid, names)
        where = f"AND feature_id IN ({', '.join(map(str, ids.values()))})"
    df = decode(tap_query(f"SELECT oid, sid, feature_id, band, version, feature_value "
                          f"FROM alerce_tap.feature WHERE oid = {oid} AND sid = {sid} {where}"))
    if wide:
        df["col"] = df.feature_name + np.where(df.band_name != "", "_" + df.band_name, "")
        return df.pivot_table(index="oid", columns="col", values="feature_value")
    return df.sort_values(["feature_name", "band"]).reset_index(drop=True)


def get_probabilities(oid, sid: int, classifier: str | None = None) -> pd.DataFrame:
    """All class probabilities of one object, decoded and sorted by classifier/rank."""
    where = "" if classifier is None else f"AND classifier_id = {classifier_id(classifier)}"
    df = tap_query(f"SELECT oid, sid, classifier_id, class_id, classifier_version, "
                   f"probability, ranking, lastmjd FROM alerce_tap.probability "
                   f"WHERE oid = {as_oid(oid)} AND sid = {sid} {where}")
    return decode(df).sort_values(["classifier_name", "ranking"]).reset_index(drop=True)


def get_magstats(oid, sid: int = 0) -> pd.DataFrame:
    """Per-band magnitude statistics (currently ZTF only)."""
    df = tap_query(f"SELECT * FROM alerce_tap.magstat WHERE oid = {as_oid(oid)} AND sid = {sid}")
    return decode(df)


def get_xmatch(oid, sid: int) -> pd.DataFrame:
    """AllWISE and Gaia DR3 counterparts stored in the database (xmatch table)."""
    return tap_query(f"""
        SELECT c.catalog_name, x.oid_catalog, x.dist,
               a.w1mpro, a.w2mpro, a.w3mpro, a.w4mpro, a.w1mpro - a.w2mpro AS w1_w2,
               a.j_m_2mass, a.h_m_2mass, a.k_m_2mass,
               g.phot_g_mean_mag, g.bp_rp, g.parallax, g.parallax_error, g.pm, g.ruwe,
               g.phot_variable_flag, g.in_qso_candidates, g.in_galaxy_candidates,
               g.classprob_dsc_combmod_quasar, g.classprob_dsc_combmod_galaxy,
               g.classprob_dsc_combmod_star
        FROM alerce_tap.xmatch AS x
        JOIN alerce_tap.catalog_id_lut AS c ON c.catid = x.catid
        LEFT JOIN alerce_tap.allwise AS a ON x.catid = 0 AND a.oid_catalog = x.oid_catalog
        LEFT JOIN alerce_tap.gaiadr3_source AS g ON x.catid = 1 AND g.oid_catalog = x.oid_catalog
        WHERE x.oid = {as_oid(oid)} AND x.sid = {sid}""")


# ---------------------------------------------------------------------------
# 7. External crossmatches: catsHTM and the experimental explorer
# ---------------------------------------------------------------------------
def catshtm_crossmatch(ra: float, dec: float, radius_arcsec: float = 5.0) -> dict:
    """Closest counterpart in each catsHTM catalog (2MASS, WISE, GALEX, XMM,
    SDSS, SpecSDSS, NEDz, Gaia, DECaLS, ...).

    Returns {catalog: {column: value}}.  Units are in ``catshtm_units``.
    """
    data = get_json(f"{CATSHTM_URL}/crossmatch_all",
                    dict(ra=round(ra, 7), dec=round(dec, 7), radius=radius_arcsec))
    out = {}
    for item in data:
        (cat, fields), = item.items()
        out[cat] = {k: v["value"] for k, v in fields.items()}
        out[cat]["_units"] = {k: v["unit"] for k, v in fields.items()}
    return out


def explorer_xmatch(oid, survey: str) -> pd.DataFrame:
    """Spectroscopic redshifts and classifications around an object, as shown in the
    experimental explorer (Simbad, NED, SDSS, DESI DR1, 2dF, 6dF, GAMA, Milliquas,
    VSX, Gaia, ...).  ``category`` is 'stellar', 'agn' or 'host'."""
    name = oid if survey != "ztf" or isinstance(oid, str) else oid_to_ztf_name(oid)
    data = get_json(f"{EXPLORER_URL}/api/xmatch_overlay", dict(oid=str(name), survey_id=survey),
                    timeout=120)
    cols = ["cat_name", "category", "name", "type", "z", "z_err", "sep", "ra", "dec"]
    df = pd.DataFrame(data.get("overlay", []))
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df.reindex(columns=cols).sort_values("sep").reset_index(drop=True)


def get_stamps(oid, sid: int, measurement_id) -> dict:
    """Science / template / difference cut-outs of one detection as 2-D arrays.

    LSST: multisurvey stamps API (51x51 pixels of 0.2"; FITS with UNCERT and PSF extensions).
    ZTF: ALeRCE's avro service (63x63 pixels of 1"; gzipped FITS).
    """
    import gzip
    import io
    from astropy.io import fits
    out = {}
    for kind in ("science", "template", "difference"):
        if sid == 0:
            url, params = "https://avro.alerce.online/get_stamp", dict(
                oid=display_name(oid, 0), candid=measurement_id, type=kind, format="fits")
        else:
            url, params = f"{API_URL}/stamps_api/stamp", dict(
                oid=oid, measurement_id=measurement_id, survey_id="lsst", file_format="fits",
                stamp_type="cutout" + kind.capitalize(), is_compressed="false")
        path = _cache_path(url + json.dumps(params, sort_keys=True, default=str), ".fits")
        if not (USE_CACHE and os.path.exists(path)):
            def _get():
                r = requests.get(url, params=params, timeout=60)
                r.raise_for_status()
                return r.content
            content = _with_retries(_get)
            if content[:2] == b"\x1f\x8b":
                content = gzip.decompress(content)
            with open(path, "wb") as f:
                f.write(content)
        with warnings.catch_warnings():  # ZTF stamps have a slightly non-standard header
            warnings.simplefilter("ignore")
            with fits.open(path) as hdul:
                out[kind] = np.array(hdul[0].data, dtype=float)
    return out


def plot_stamps(stamps: dict, title: str = "", axes=None):
    """Show science, template and difference cut-outs side by side."""
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, (kind, img) in zip(axes, stamps.items()):
        img = np.nan_to_num(img)
        lo, _ = ZScaleInterval().get_limits(img)
        norm = ImageNormalize(img, vmin=lo, vmax=np.percentile(img, 99.8), stretch=AsinhStretch(0.05))
        ax.imshow(img, origin="lower", cmap="gray", norm=norm)
        ax.set_title(f"{title} {kind}".strip(), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    return axes


def explorer_url(oid, survey: str) -> str:
    """Link to the object in the experimental explorer (shows ZTF + LSST together)."""
    if survey == "ztf" and not isinstance(oid, str):
        oid = oid_to_ztf_name(oid)
    return f"{EXPLORER_URL}/?survey={survey}&oid={oid}"


def show_link(oid, survey: str, text: str | None = None):
    from IPython.display import HTML, display
    url = explorer_url(oid, survey)
    display(HTML(f'<a href="{url}" target="_blank">{text or url}</a>'))


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------
def plot_lightcurve(lc: pd.DataFrame, flux: str = "diff", mag: bool = False, ax=None,
                    title: str | None = None, show_forced: bool = True):
    """Plot one or several light curves from ``get_lightcurve`` (concatenate ZTF and
    LSST frames to overplot them).  ``flux`` is 'diff' or 'sci'.  Circles = ZTF,
    squares = LSST; filled = detections, open = forced photometry."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    y = f"{flux}_mag" if mag else f"{flux}_flux"
    for (survey, band, kind), g in lc.groupby(["survey", "band_name", "kind"]):
        if kind == "forced" and not show_forced:
            continue
        g = g[np.isfinite(g[y])]
        if g.empty:
            continue
        color = BAND_COLORS.get(band, "k")
        ax.errorbar(g.mjd, g[y], yerr=g[y + "_err"], fmt=SURVEY_MARKERS[survey], ms=4,
                    color=color, mfc=color if kind == "det" else "none", alpha=0.8,
                    elinewidth=0.6, label=f"{survey.upper()} {band} {kind}")
    if mag:
        ax.invert_yaxis()
        ax.set_ylabel(f"{'difference' if flux == 'diff' else 'total'} AB mag")
    else:
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_ylabel(f"{'difference' if flux == 'diff' else 'total'} flux [nJy]")
    ax.set_xlabel("MJD")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if title:
        ax.set_title(title)
    return ax


def phase_fold(mjd, period: float, t0: float = 0.0):
    return ((np.asarray(mjd) - t0) / period) % 1.0


def plot_folded(lc: pd.DataFrame, period: float, flux: str = "sci", ax=None, title=None):
    """Phase-folded light curve (two cycles), magnitudes, ZTF and LSST overplotted."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    y = f"{flux}_mag"
    for (survey, band), g in lc.groupby(["survey", "band_name"]):
        g = g[np.isfinite(g[y])]
        ph = phase_fold(g.mjd, period)
        color = BAND_COLORS.get(band, "k")
        for shift in (0, 1):
            ax.plot(ph + shift, g[y], SURVEY_MARKERS[survey], ms=3, color=color, alpha=0.6,
                    label=f"{survey.upper()} {band}" if shift == 0 else None)
    ax.invert_yaxis()
    ax.set_xlabel(f"phase (P = {period:.6f} d)")
    ax.set_ylabel("AB mag")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if title:
        ax.set_title(title)
    return ax

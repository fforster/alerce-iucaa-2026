"""
alerce_ms.py — small helper library for the ALeRCE multisurvey tutorial (IUCAA 2026).

Everything here is a thin wrapper around three public services:

* TAP    https://tap.alerce.online/tap           (schema ``alerce_tap``, ADQL queries)
* API    https://api-lsst.alerce.online/<name>_api (multisurvey REST API, JSON)
* Labs   https://explorer-experiment.alerce.online (experimental explorer)

plus the ALeRCE catsHTM crossmatch service (https://catshtm.alerce.online) and, for the
catalog crossmatch, VizieR, SIMBAD (each with mirrors) and NED.

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
import io
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
ALERCE_EXPLORER_URL = "https://lsst.alerce.online"
CATSHTM_URL = "https://catshtm.alerce.online"

SID = {"ztf": 0, "lsst": 1, "lsst_ss": 2}
SURVEY = {0: "ztf", 1: "lsst", 2: "lsst"}

# ALeRCE band colours, as in the ALeRCE explorers: the Okabe-Ito colour-blind-friendly palette
# (Okabe & Ito 2008, https://jfly.uni-koeln.de/color/; recommended in Nature Methods by
# Wong 2011, Nat. Methods 8, 441, doi:10.1038/nmeth.1618).
# ZTF and LSST share g, r, i: the same band gets the same colour, surveys differ by marker.
BAND_COLORS = {
    "u": "#56B4E9",  # sky blue
    "g": "#009E73",  # bluish green
    "r": "#D55E00",  # vermillion
    "i": "#E69F00",  # orange
    "z": "#CC79A7",  # reddish purple
    "y": "#0072B2",  # blue
}
# ALeRCE drawing order when many bands overlap: g on top, then r, i, z, u, and y at the bottom
BAND_ZORDER = {"u": 2, "g": 6, "r": 5, "i": 4, "z": 3, "y": 1}
BAND_ORDER = ["u", "g", "r", "i", "z", "y"]  # increasing wavelength (legend order)
SURVEY_ORDER = ["ztf", "lsst"]
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
# 7. External crossmatches: catsHTM, and the explorer's catalog crossmatch
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


# ---------------------------------------------------------------------------
# The experimental explorer's crossmatch, computed on your own computer.
#
# Same catalogs, radii, quality cuts and categories as the explorer
# (src/services/xmatch.py of alerce-explorer-experimental), but queried directly:
# VizieR cone searches (with mirrors), SIMBAD TAP (with mirrors) and NED TAP.
# This keeps the load off the explorer's small server.
# ---------------------------------------------------------------------------
C_KMS = 299792.458
XM_RADIUS = dict(stellar=3.0, agn=3.0, sdss=30.0, desi=30.0, zcat=36.0, simbad=36.0, ned=36.0, host_ext=60.0)


def _num(value):
    """float, or None for masked / empty / sentinel values."""
    if value is None or value is np.ma.masked:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) or value in (-99.0, -999.0, -9999.0) else value


def _str(value):
    if value is None or value is np.ma.masked:
        return None
    s = (value.decode() if isinstance(value, bytes) else str(value)).strip()
    return s if s not in ("", "--", "nan") else None


def _position(r):
    """The catalog's own coordinates (as the explorer uses them), else VizieR's computed J2000 position."""
    for ra_col, dec_col in [("RA_ICRS", "DE_ICRS"), ("RAICRS", "DEICRS"), ("RAJ2000", "DEJ2000")]:
        if ra_col in r.colnames and _num(r[ra_col]) is not None and _num(r[dec_col]) is not None:
            return _num(r[ra_col]), _num(r[dec_col])
    return _num(r["_RAJ2000"]), _num(r["_DEJ2000"])


def _designation(cat_name, ra, dec):
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra, dec, unit="deg")
    return f"{cat_name} J" + c.to_string("hmsdms", sep="", precision=1).replace(" ", "")


_MQ_CLASS = {"Q": "QSO", "A": "AGN", "B": "BL Lac", "K": "radio QSO", "N": "narrow AGN", "L": "lensed QSO"}
_VV_CLASS = {"Q": "QSO", "A": "AGN", "B": "BL Lac"}


def _row_gaia(r):
    return dict(name=(_str(r["DR3Name"]) or "").replace("Gaia DR3 ", ""),
                type="variable star" if (_str(r["VarFlag"]) or "").upper() == "VARIABLE" else "star")


def _row_vsx(r):
    return dict(name=_str(r["Name"]), type=_str(r["Type"])) if _str(r["Type"]) else None


def _row_milliquas(r):
    t = _str(r["Type"]) or ""
    return dict(name=_str(r["Name"]), type=_MQ_CLASS.get(t[:1], t[:1] or "AGN"), z=_num(r["z"]))


def _row_veron(r):
    return dict(name=_str(r["Name"]), type=_VV_CLASS.get((_str(r["Cl"]) or "")[:1], "AGN"), z=_num(r["z"]))


def _row_hyperleda(r):
    return dict(name=_str(r["ANames"]), type=_str(r["OType"]) or "G")


def _row_sdss(r):     # galaxies only; zsp is the spectroscopic redshift
    if _num(r["class"]) != 3:
        return None
    return dict(name=str(r["objID"]), type="GALAXY", z=_num(r["zsp"]), z_err=_num(r["e_zsp"]))


def _row_desi(r):     # reliable redshifts (ZWARN = 0), no stars
    if _num(r["ZWARN"]) != 0 or (_str(r["OType"]) or "").upper() == "STAR":
        return None
    ra, dec = _position(r)
    return dict(name=f"DESI J{ra:08.4f}{dec:+08.4f}", type=_str(r["OType"]),
                z=_num(r["z"]), z_err=_num(r["e_z"]))


def _zcat(name, z, types=(), ez=None, cz=False, keep=lambda r: True):
    """Row parser for a spectroscopic-redshift catalog (cz in km/s if cz=True)."""
    def parse(r):
        if not keep(r) or _num(r[z]) is None:
            return None
        zz, ee = _num(r[z]), _num(r[ez]) if ez else None
        if cz:
            zz, ee = zz / C_KMS, (ee / C_KMS if ee is not None else None)
        label = " / ".join(s for s in (_str(r[c]) for c in types) if s)
        return dict(name=_designation(name, *_position(r)), type=label or None, z=zz, z_err=ee)
    return parse


def _q(col, minimum):
    return lambda r: (_num(r[col]) or 0) >= minimum


# VizieR table -> (catalog name, category, search radius [arcsec], row parser)
XM_VIZIER = {
    "I/355/gaiadr3": ("Gaia DR3", "stellar", XM_RADIUS["stellar"], _row_gaia),
    "B/vsx/vsx": ("VSX", "stellar", XM_RADIUS["stellar"], _row_vsx),
    "VII/294/catalog": ("Milliquas", "agn", XM_RADIUS["agn"], _row_milliquas),
    "VII/258/vv10": ("Veron-Cetty", "agn", XM_RADIUS["agn"], _row_veron),
    "V/154/sdss16": ("SDSS", "host", XM_RADIUS["sdss"], _row_sdss),
    "V/161/zcatdr1": ("DESI", "host", XM_RADIUS["desi"], _row_desi),
    "VII/289/dr16q": ("SDSS DR16 QSO", "agn", XM_RADIUS["zcat"], _zcat("SDSS DR16 QSO", "z", ["Class"])),
    "VII/259/6dfgs": ("6dFGS", "host", XM_RADIUS["zcat"], _zcat("6dFGS", "cz", ez="e_cz", cz=True, keep=_q("q_cz", 3))),
    "J/MNRAS/513/439/gamadr4": ("GAMA DR4", "host", XM_RADIUS["zcat"],
                                _zcat("GAMA DR4", "z", ["Survey"],
                                      keep=lambda r: _q("q_z", 3)(r) and _str(r["IsBest"]) == "1")),
    "J/ApJS/199/26/table3": ("2MRS", "host", XM_RADIUS["zcat"], _zcat("2MRS", "cz", ["type"], cz=True)),
    "J/MNRAS/474/4151/wigglez": ("WiggleZ", "host", XM_RADIUS["zcat"], _zcat("WiggleZ", "z", ["Class"], keep=_q("q_z", 3))),
    "J/ApJS/184/218/table3": ("zCOSMOS", "host", XM_RADIUS["zcat"], _zcat("zCOSMOS", "z", keep=_q("CClass", 2.5))),
    "J/A+A/609/A84/vipersw1": ("VIPERS PDR2", "host", XM_RADIUS["zcat"],
                               _zcat("VIPERS PDR2", "zsp", ["classFlag"], keep=_q("zflg", 2))),
    "J/A+A/609/A84/vipersw4": ("VIPERS PDR2", "host", XM_RADIUS["zcat"],
                               _zcat("VIPERS PDR2", "zsp", ["classFlag"], keep=_q("zflg", 2))),
    "J/MNRAS/472/273/ozdesdr1": ("OzDES DR1", "host", XM_RADIUS["zcat"],
                                 _zcat("OzDES DR1", "z", ["types"], keep=lambda r: _num(r["Flag"]) in (3, 4))),
    "VII/250/2dfgrs": ("2dFGRS", "host", XM_RADIUS["zcat"], _zcat("2dFGRS", "z", keep=_q("q_z", 3))),
    "VII/281/glade2": ("GLADE v2", "host", XM_RADIUS["zcat"],
                       _zcat("GLADE v2", "z", ["Flag1"],
                             keep=lambda r: _str(r["Flag2"]) == "2" and _str(r["Flag1"]) in ("G", "Q"))),
    "J/MNRAS/548/G522/hecatev2": ("HECATE", "host", XM_RADIUS["zcat"],
                                  lambda r: None if _num(r["HRV"]) is None else dict(
                                      name=_str(r["OBJNAME"]) or _designation("HECATE", *_position(r)),
                                      type=_str(r["ActivClass"]), z=_num(r["HRV"]) / C_KMS,
                                      z_err=(_num(r["e_HRV"]) / C_KMS) if _num(r["e_HRV"]) else None)),
    "VII/237/pgc": ("HyperLEDA", "host", XM_RADIUS["host_ext"], _row_hyperleda),
}


def _simbad_category(otype):
    """Route a SIMBAD object type to stellar / agn / host, as the explorer does."""
    s = (otype or "").strip().lower()
    if (any(k in s for k in ("qso", "quasar", "seyfert", "blazar", "bl_lac", "bllac", "liner"))
            or s in ("agn", "sy1", "sy2", "sy", "bla", "bll")):
        return "agn"
    if "galax" in s or s in ("g", "gig", "gic", "gip", "ig", "grg", "clg", "gg",
                             "gpair", "ggroup", "emg", "rg", "brg", "blg", "sbg", "h2g"):
        return "host"
    return "stellar"


def _is_transient_type(t):
    """A catalogued supernova: its redshift is the event's own, not a host galaxy's."""
    s = t.strip().upper() if isinstance(t, str) else ""
    return "SUPERNOV" in s or s in ("SN", "SN?", "SN*", "SLSN", "SLSN-I", "SLSN-II")


def _xm_vizier(ra, dec):
    """All the VizieR catalogs, one cone search per search radius."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    coord, rows, failed = SkyCoord(ra, dec, unit="deg"), [], []
    for radius in sorted({v[2] for v in XM_VIZIER.values()}):
        cats = [c for c, v in XM_VIZIER.items() if v[2] == radius]
        try:
            with warnings.catch_warnings():    # astroquery's VOTable parsing warns about empty cells
                warnings.simplefilter("ignore")
                res = vizier_query_region(coord, radius * u.arcsec, cats, columns=("**", "_RAJ2000", "_DEJ2000"),
                                          row_limit=200, timeout=60)
        except Exception as e:
            failed.append(f"VizieR {radius:g}\" ({type(e).__name__})")
            continue
        for key in res.keys():
            cat_name, category, _, parse = XM_VIZIER[key]
            for r in res[key]:
                info = parse(r)
                if info:
                    ra_r, dec_r = _position(r)
                    rows.append(dict(cat_name=cat_name, category=category, ra=ra_r, dec=dec_r, **info))
    return rows, failed


def _xm_simbad(ra, dec):
    radius = XM_RADIUS["simbad"] / 3600
    df = simbad_tap(f"""SELECT b.main_id, b.ra, b.dec, b.otype, d.label, b.rvz_redshift, b.rvz_err, b.rvz_type
        FROM basic AS b LEFT JOIN otypedef AS d ON b.otype = d.otype
        WHERE 1 = CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))""")
    rows = []
    for r in df.itertuples():
        otype = r.label if isinstance(r.label, str) else r.otype
        z_err = _num(r.rvz_err)
        if z_err is not None and r.rvz_type == "v":     # error given as a velocity
            z_err /= C_KMS
        rows.append(dict(cat_name="Simbad", category=_simbad_category(otype), name=r.main_id, type=otype,
                         z=_num(r.rvz_redshift), z_err=z_err, ra=r.ra, dec=r.dec))
    return rows


def _xm_ned(ra, dec):
    radius = XM_RADIUS["ned"] / 3600
    query = f"""SELECT prefname, ra, dec, z, zunc, prefphytype FROM NEDTAP.objdir
        WHERE CONTAINS(POINT('J2000', ra, dec), CIRCLE('J2000', {ra}, {dec}, {radius})) = 1"""
    for attempt in range(3):          # NED drops connections now and then
        try:
            r = requests.get("https://ned.ipac.caltech.edu/tap/sync", timeout=60,
                             params=dict(request="doQuery", lang="adql", format="csv", query=query))
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5)
    df = pd.read_csv(io.StringIO(r.text))
    return [dict(cat_name="NED", category="host", name=row.prefname, type=_str(row.prefphytype),
                 z=_num(row.z), z_err=_num(row.zunc), ra=row.ra, dec=row.dec) for row in df.itertuples()]


def catalog_xmatch(ra: float, dec: float, cache: bool = True) -> pd.DataFrame:
    """Stars, AGN and host-galaxy redshifts around (ra, dec): the crossmatch of the experimental explorer
    (SIMBAD, NED, Gaia DR3, VSX, Milliquas, Véron-Cetty, SDSS, DESI DR1, 6dF, 2dF, GAMA, 2MRS, WiggleZ, zCOSMOS,
    VIPERS, OzDES, GLADE, HECATE, HyperLEDA), computed here by querying the catalogs directly.

    ``category`` is 'stellar', 'agn' or 'host'.  Stars and AGN must be within 3" (the transient *is* the star or
    AGN); host galaxies are searched within 30-60" and are only kept if they have a redshift.  ``sep`` is in arcsec.
    """
    cols = ["cat_name", "category", "name", "type", "z", "z_err", "sep", "ra", "dec"]
    path = _cache_path(f"catalog_xmatch {ra:.6f} {dec:.6f}", ".pkl")
    if cache and USE_CACHE and os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as pool:
        jobs = {"VizieR": pool.submit(_xm_vizier, ra, dec), "SIMBAD": pool.submit(_xm_simbad, ra, dec),
                "NED": pool.submit(_xm_ned, ra, dec)}
    rows, failed = [], []
    for service, job in jobs.items():
        try:
            out = job.result()
        except Exception as e:
            failed.append(f"{service} ({type(e).__name__})")
            continue
        if service == "VizieR":
            out, bad = out
            failed += bad
        rows += out
    df = pd.DataFrame(rows).reindex(columns=cols)
    if not df.empty:
        df = df.dropna(subset=["ra", "dec"])
        df["sep"] = angular_sep_arcsec(ra, dec, df.ra.values, df.dec.values)
        point_source = df.category.isin(["stellar", "agn"])
        host_ok = df.z.notna() & ~df.type.map(_is_transient_type)
        df = df[(point_source & (df.sep <= XM_RADIUS["stellar"])) | (~point_source & host_ok)]
        order = df.category.map({"stellar": 0, "agn": 1, "host": 2})
        df = df.assign(_o=order).sort_values(["_o", "sep"]).drop(columns="_o").reset_index(drop=True)
    if failed:
        print("Not answered (try again later):", ", ".join(failed))
    else:
        df.to_pickle(path)        # only cache complete answers
    return df


SNAPSHOT_URL = "https://raw.githubusercontent.com/fforster/alerce-iucaa-2026/main/data/explorer_xmatch_snapshot.json"


@functools.lru_cache(maxsize=1)
def _explorer_snapshot() -> dict:
    """Explorer crossmatches saved before the CDS outage of November 2026 (for the tutorial objects)."""
    if os.path.exists("data/explorer_xmatch_snapshot.json"):
        with open("data/explorer_xmatch_snapshot.json") as f:
            return json.load(f)
    try:
        return get_json(SNAPSHOT_URL, timeout=30)
    except Exception:
        return {}


def object_xmatch(oid, survey: str) -> pd.DataFrame:
    """``catalog_xmatch`` at the mean position of an ALeRCE object (ZTF name or oid, or LSST oid).

    If the catalog services answer with fewer matches than the explorer gave before the CDS outage of November
    2026 (saved for the tutorial objects in ``data/explorer_xmatch_snapshot.json``), the saved answer is shown."""
    name = oid if survey != "ztf" or isinstance(oid, str) else oid_to_ztf_name(oid)
    obj = get_object(name, survey)
    df = catalog_xmatch(float(obj["meanra"]), float(obj["meandec"]))
    snap = _explorer_snapshot().get(f"{survey}:{name}")
    if snap and len(snap["overlay"]) > len(df):
        print(f"(Only {len(df)} matches now, probably because of the CDS outage; showing the "
              f"{len(snap['overlay'])} matches the explorer gave on {snap['saved']}.)")
        df = pd.DataFrame(snap["overlay"]).reindex(columns=df.columns).sort_values("sep").reset_index(drop=True)
    return df


VIZIER_SERVERS = ["vizier.cds.unistra.fr", "vizier.idia.ac.za", "vizier.cfa.harvard.edu"]
_vizier_failed = set()


def vizier_query_region(coord, radius, catalog, columns=("**", "_r"), row_limit=50, timeout=20):
    """astroquery VizieR cone search that tries CDS first and then its mirrors (IDIA in South Africa, CfA),
    remembering servers that did not answer. Returns an astroquery TableList."""
    from astroquery.vizier import Vizier
    last = None
    for server in [s for s in VIZIER_SERVERS if s not in _vizier_failed] or VIZIER_SERVERS:
        try:
            v = Vizier(columns=list(columns), row_limit=row_limit, vizier_server=server, timeout=timeout)
            return v.query_region(coord, radius=radius, catalog=catalog)
        except Exception as e:
            _vizier_failed.add(server)
            last = e
    raise last



SIMBAD_TAP_SERVERS = ["https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
                      "https://simbad.idia.ac.za/simbad/sim-tap/sync",       # mirror in South Africa
                      "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync"]  # mirror at CfA (USA)
_simbad_failed = set()


def simbad_tap(query: str, timeout: float = 30) -> pd.DataFrame:
    """Run an ADQL query on SIMBAD's TAP service, trying CDS first and then its mirrors (IDIA, CfA),
    remembering servers that did not answer. ``df.attrs["server"]`` says which one answered."""
    last = None
    for url in [u for u in SIMBAD_TAP_SERVERS if u not in _simbad_failed] or SIMBAD_TAP_SERVERS:
        try:
            r = requests.get(url, params=dict(request="doQuery", lang="adql", format="csv", query=query),
                             timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.attrs["server"] = url.split("/")[2]
            return df
        except Exception as e:
            _simbad_failed.add(url)
            last = e
    raise last


def simbad_query_region(ra, dec, radius_arcsec=2, timeout=20):
    """SIMBAD cone search: main identifier, object type, redshift and distance in arcsec (``simbad_tap``)."""
    return simbad_tap(f"""SELECT main_id, otype, rvz_redshift,
        DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) * 3600 AS dist_arcsec
        FROM basic WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600}))
        ORDER BY dist_arcsec""", timeout=timeout)


def _cd_matrix(header):
    """2x2 CD matrix [deg/pixel] from a FITS header (CD_ij, or PC_ij scaled by CDELT_i)."""
    if "CD1_1" in header:
        return np.array([[header["CD1_1"], header.get("CD1_2", 0.0)],
                         [header.get("CD2_1", 0.0), header["CD2_2"]]])
    if "PC1_1" in header:
        c1, c2 = header.get("CDELT1", 1.0), header.get("CDELT2", 1.0)
        return np.array([[c1 * header["PC1_1"], c1 * header.get("PC1_2", 0.0)],
                         [c2 * header.get("PC2_1", 0.0), c2 * header["PC2_2"]]])
    return None


def north_up(image, header):
    """Resample a stamp so that North is up and East is left (for ``imshow(origin="lower")``).

    The CD matrix maps a pixel step (dx, dy) to a step on the sky (dxi towards East, deta towards North), so its
    inverse tells us which pixel direction points North and which points East. We then build an output grid whose
    "up" is North and whose "right" is West, centred on the alert pixel (CRPIX), and interpolate the image onto it.
    This removes any rotation and mirror flip. Pixel scale and image size are kept; corners that fall outside the
    original cut-out become NaN. Same idea as the experimental explorer (stamps.js), which rotates the canvas.

    Returns ``image`` unchanged if the header has no WCS.
    """
    from scipy.ndimage import affine_transform
    cd = _cd_matrix(header)
    if cd is None or abs(np.linalg.det(cd)) < 1e-20:
        return image
    inv = np.linalg.inv(cd)
    north = inv @ [0.0, 1.0]
    east = inv @ [1.0, 0.0]
    north, west = north / np.hypot(*north), -east / np.hypot(*east)
    ny, nx = image.shape
    # array index order is (row = y, col = x); output (row, col) offsets = (north, west) steps
    matrix = np.array([[north[1], west[1]],
                       [north[0], west[0]]])
    center_in = np.array([header.get("CRPIX2", (ny + 1) / 2) - 1, header.get("CRPIX1", (nx + 1) / 2) - 1])
    center_out = np.array([(ny - 1) / 2, (nx - 1) / 2])
    offset = center_in - matrix @ center_out
    return affine_transform(np.asarray(image, float), matrix, offset=offset, order=1,
                            mode="constant", cval=np.nan)


ZTF_PIXEL_ARCSEC = 1.012   # ZTF pixel scale
STAMP_KINDS = ("science", "template", "difference")


def _stamp_fits(oid, sid: int, measurement_id, kind: str):
    """Download (once, then cache) one cut-out and return (image, header)."""
    import gzip
    from astropy.io import fits
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
            return np.array(hdul[0].data, dtype=float), hdul[0].header.copy()


def get_stamps(oid, sid: int, measurement_id, north: bool = True) -> dict:
    """Science / template / difference cut-outs of one detection as 2-D arrays.

    LSST: multisurvey stamps API (51x51 pixels of 0.2"; FITS with a full WCS, UNCERT and PSF extensions).
    ZTF: ALeRCE's avro service (63x63 pixels of 1"; gzipped FITS without WCS).

    With ``north=True`` (default) the stamps are oriented North up, East left when shown with
    ``imshow(origin="lower")``: LSST stamps are resampled with their WCS (``north_up``); ZTF cut-outs are already
    aligned with the sky (RA and Dec both decrease with pixel index), so their rows are just flipped.
    """
    out = {}
    for kind in STAMP_KINDS:
        img, header = _stamp_fits(oid, sid, measurement_id, kind)
        if north:
            img = np.flipud(img) if sid == 0 else north_up(img, header)
        out[kind] = img
    return out


def _tan_pixel_to_sky(x, y, header):
    """FITS (1-based) pixel -> (ra, dec) [deg] with the gnomonic (TAN) projection of the header's WCS."""
    cd = _cd_matrix(header)
    xi, eta = np.radians(cd @ [x - header["CRPIX1"], y - header["CRPIX2"]])
    a0, d0 = np.radians(header["CRVAL1"]), np.radians(header["CRVAL2"])
    rho = np.hypot(xi, eta)
    if rho == 0:
        return header["CRVAL1"], header["CRVAL2"]
    c = np.arctan(rho)
    dec = np.arcsin(np.cos(c) * np.sin(d0) + eta * np.sin(c) * np.cos(d0) / rho)
    ra = a0 + np.arctan2(xi * np.sin(c), rho * np.cos(d0) * np.cos(c) - eta * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360, np.degrees(dec)


def stamp_center(oid, sid: int, measurement_id):
    """(ra, dec) [deg] at the centre of a cut-out: the WCS reference point for LSST, the detection position
    (from the ``detection`` table) for ZTF, whose cut-outs carry no WCS."""
    if sid != 0:
        _, h = _stamp_fits(oid, sid, measurement_id, "science")
        return h["CRVAL1"], h["CRVAL2"]
    row = tap_query(f"SELECT ra, dec FROM alerce_tap.detection WHERE oid = {as_oid(oid)} AND sid = 0 "
                    f"AND measurement_id = {int(measurement_id)}")
    return float(row.ra.iloc[0]), float(row.dec.iloc[0])


def stamp_corners(oid, sid: int, measurement_id) -> np.ndarray:
    """(ra, dec) [deg] of the 4 corners of a cut-out: from the WCS for LSST; for ZTF from the detection position
    and the 1.012"/pixel scale (ZTF cut-outs are aligned with the sky)."""
    img, h = _stamp_fits(oid, sid, measurement_id, "science")
    ny, nx = img.shape
    if sid != 0:
        return np.array([_tan_pixel_to_sky(x, y, h) for x, y in
                         ((0.5, 0.5), (nx + 0.5, 0.5), (nx + 0.5, ny + 0.5), (0.5, ny + 0.5))])
    ra0, dec0 = stamp_center(oid, sid, measurement_id)
    half = np.array([nx, ny]) / 2 * ZTF_PIXEL_ARCSEC / 3600
    return np.array([(ra0 + sx * half[0] / np.cos(np.radians(dec0)), dec0 + sy * half[1])
                     for sx, sy in ((1, -1), (-1, -1), (-1, 1), (1, 1))])


def draw_footprint(ax, corners, center, shape, scale_arcsec, **kwargs):
    """Outline a sky polygon ``corners`` [(ra, dec), ...] on a North-up stamp of ``shape`` pixels of
    ``scale_arcsec``, centred on ``center`` = (ra, dec). East is left, so x decreases with RA."""
    ra0, dec0 = center
    ny, nx = shape
    corners = np.asarray(corners)
    dx = -(corners[:, 0] - ra0) * np.cos(np.radians(dec0)) * 3600 / scale_arcsec
    dy = (corners[:, 1] - dec0) * 3600 / scale_arcsec
    x, y = (nx - 1) / 2 + dx, (ny - 1) / 2 + dy
    style = dict(color="#56B4E9", lw=1.2)
    style.update(kwargs)
    return ax.plot(np.append(x, x[0]), np.append(y, y[0]), **style)


def plot_stamps(stamps: dict, title: str = "", axes=None, compass: bool = True, footprint: dict | None = None):
    """Show science, template and difference cut-outs side by side (North up, East left as returned by
    ``get_stamps``); ``compass`` draws N and E arrows on the first panel.

    ``footprint`` (optional) outlines another cut-out on every panel: a dict with keys ``corners`` (sky corners,
    see ``stamp_corners``), ``center`` (sky position of this stamp's centre, see ``stamp_center``),
    ``scale_arcsec`` (pixel scale of this stamp) and optionally ``label`` and plot keywords in ``style``.
    """
    import copy
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(8, 3))
    cmap = copy.copy(plt.get_cmap("gray"))
    cmap.set_bad("black")
    for k, (ax, (kind, img)) in enumerate(zip(axes, stamps.items())):
        good = img[np.isfinite(img)]
        lo, _ = ZScaleInterval().get_limits(good)
        norm = ImageNormalize(vmin=lo, vmax=np.percentile(good, 99.8), stretch=AsinhStretch(0.05))
        ax.imshow(np.ma.masked_invalid(img), origin="lower", cmap=cmap, norm=norm)
        ax.set_title(f"{title} {kind}".strip(), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if footprint is not None:
            ax.autoscale(False)
            draw_footprint(ax, footprint["corners"], footprint["center"], img.shape, footprint["scale_arcsec"],
                           **footprint.get("style", {}))
            if k == 0 and footprint.get("label"):
                ax.text(0.03, 0.97, f"□ {footprint['label']}", transform=ax.transAxes, va="top", fontsize=8,
                        color=footprint.get("style", {}).get("color", "#56B4E9"),
                        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.5))
        if compass and k == 0:
            x0, y0, length = 0.88, 0.08, 0.2  # axes fraction
            for dx, dy, label in ((0, length, "N"), (-length, 0, "E")):
                ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.2))
                ax.text(x0 + 1.25 * dx, y0 + 1.25 * dy, label, transform=ax.transAxes, color="#ffd166",
                        fontsize=8, ha="center", va="center")
    return axes


def explorer_url(oid, survey: str, site: str = "experimental") -> str:
    """Link to the object in an ALeRCE explorer.

    site="experimental": the experimental explorer (ALeRCE Labs), which shows ZTF + LSST together;
    site="alerce": the ALeRCE multisurvey explorer at lsst.alerce.online.
    """
    if survey == "ztf" and not isinstance(oid, str):
        oid = oid_to_ztf_name(oid)
    if site == "alerce":
        return f"{ALERCE_EXPLORER_URL}/object/{oid}?survey={survey}"
    return f"{EXPLORER_URL}/?survey={survey}&oid={oid}"


def explorer_urls(oid, survey: str) -> dict:
    """Links to the object in both explorers: {"ALeRCE explorer": ..., "experimental explorer": ...}."""
    return {"ALeRCE explorer": explorer_url(oid, survey, "alerce"),
            "experimental explorer": explorer_url(oid, survey, "experimental")}


def show_link(oid, survey: str, text: str | None = None):
    """Display one line with links to the object in both explorers (lsst.alerce.online and the experimental one)."""
    from IPython.display import HTML, display
    if text is None:
        text = display_name(as_oid(oid), SID[survey]) if survey in SID else str(oid)
    links = " · ".join(f'<a href="{url}" target="_blank">{site}</a>' for site, url in explorer_urls(oid, survey).items())
    display(HTML(f"<b>{text}</b>: {links}"))


def explorer_links(oid, survey: str) -> str:
    """Short HTML links to the object in both explorers, for use inside tables."""
    short = {"ALeRCE explorer": "ALeRCE", "experimental explorer": "experimental"}
    return " · ".join(f'<a href="{url}" target="_blank">{short[site]}</a>'
                      for site, url in explorer_urls(oid, survey).items())


def show_table(df: pd.DataFrame, links: dict | None = None):
    """Display a table with clickable explorer links.

    ``links`` maps a new column name to (oid column, survey), e.g. {"ZTF": ("ztf_oid", "ztf")}; the survey can also be
    the name of a column holding "ztf" or "lsst" for each row.
    """
    from IPython.display import HTML, display
    out = df.copy()
    for name, (col, survey) in (links or {}).items():
        surveys = out[survey] if survey in out.columns else [survey] * len(out)
        out[name] = [explorer_links(v, s) if pd.notna(v) else "" for v, s in zip(out[col], surveys)]
    display(HTML(out.to_html(escape=False, index=False)))


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------
def sky_xy(ra, dec):
    """(x, y) in radians for a Mollweide axis, with RA increasing to the left (east left, as on the sky)."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    return -np.radians((ra + 180) % 360 - 180), np.radians(dec)


def sky_axes(figsize=(9, 4.8), galactic_plane=True):
    """A Mollweide all-sky axis (RA 0° in the centre, increasing to the left) for ``ax.scatter(*sky_xy(ra, dec))``.
    The Galactic plane is drawn as a grey line, with the Galactic centre marked."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")
    ticks = np.arange(-150, 151, 30)
    ax.set_xticks(np.radians(ticks))
    ax.set_xticklabels([f"{(-t) % 360}°" for t in ticks], fontsize=8, color="0.4")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    if galactic_plane:
        from astropy.coordinates import SkyCoord
        gp = SkyCoord(l=np.linspace(0, 360, 721), b=np.zeros(721), unit="deg", frame="galactic").icrs
        x, y = sky_xy(gp.ra.deg, gp.dec.deg)
        order = np.argsort(x)
        ax.plot(x[order], y[order], color="0.6", lw=1, zorder=0, label="Galactic plane")
        gc = SkyCoord(l=0, b=0, unit="deg", frame="galactic").icrs
        ax.plot(*sky_xy(gc.ra.deg, gc.dec.deg), "x", color="0.6", ms=6, zorder=0)
    return fig, ax


def band_key(band: str) -> int:
    """Position of a band in wavelength order (unknown bands go last)."""
    return BAND_ORDER.index(band) if band in BAND_ORDER else len(BAND_ORDER)


def _legend_key(label: str):
    """Sort key for a legend label such as 'LSST g det': band (wavelength), survey, detections first."""
    tokens = label.replace("(", " ").replace(")", " ").split()
    band = next((t for t in tokens if t in BAND_ORDER), None)
    upper = label.upper()
    survey = next((k for k, s in enumerate(SURVEY_ORDER) if s.upper() in upper), len(SURVEY_ORDER))
    return (band_key(band), survey, "FORCED" in upper, label)


def sort_legend(ax, **kwargs):
    """Draw the legend of ``ax`` with its entries sorted by wavelength (u, g, r, i, z, y)."""
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: _legend_key(labels[k]))
    return ax.legend([handles[k] for k in order], [labels[k] for k in order], **kwargs)


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
        # ALeRCE z-order per band; within a band, detections above forced photometry
        zorder = BAND_ZORDER.get(band, 0) + (0.5 if kind == "det" else 0)
        ax.errorbar(g.mjd, g[y], yerr=g[y + "_err"], fmt=SURVEY_MARKERS[survey], ms=4,
                    color=color, mfc=color if kind == "det" else "none", alpha=0.8,
                    elinewidth=0.6, zorder=zorder, label=f"{survey.upper()} {band} {kind}")
    if mag:
        ax.invert_yaxis()
        ax.set_ylabel(f"{'difference' if flux == 'diff' else 'total'} AB mag")
    else:
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_ylabel(f"{'difference' if flux == 'diff' else 'total'} flux [nJy]")
    ax.set_xlabel("MJD")
    sort_legend(ax, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
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
                    zorder=BAND_ZORDER.get(band, 0),
                    label=f"{survey.upper()} {band}" if shift == 0 else None)
    ax.invert_yaxis()
    ax.set_xlabel(f"phase (P = {period:.6f} d)")
    ax.set_ylabel("AB mag")
    sort_legend(ax, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if title:
        ax.set_title(title)
    return ax

# ALeRCE multisurvey tutorial — IUCAA 2026

Hands-on notebooks to do science with the [ALeRCE](https://alerce.science) broker using **ZTF and LSST alerts
together**, through the new ALeRCE **multisurvey database**, its **TAP** service, the **multisurvey REST API** and
the **experimental explorer** (ALeRCE Labs). Total time: about 3 hours.

| # | Notebook | Topic | Time | Colab |
|---|---|---|---|---|
| 00 | [Introduction](00_introduction.ipynb) | ALeRCE, ZTF and LSST; how to access the data | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/00_introduction.ipynb) |
| 01 | [The multisurvey database](01_multisurvey_database.ipynb) | all 31 tables, look-up tables, joining rules, helper functions | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/01_multisurvey_database.ipynb) |
| 02 | [One source, two surveys](02_one_source_two_surveys.ipynb) | cross-survey counterparts, the REST API, light curves, classes, stamps, explorer | 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/02_one_source_two_surveys.ipynb) |
| 03 | [Supernovae](03_supernovae.ipynb) | DELIGHT host galaxies, spectroscopic redshifts, absolute magnitudes | 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/03_supernovae.ipynb) |
| 04 | [AGN](04_agn.ipynb) | multiwavelength crossmatches, SEDs, redshifts, variability | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/04_agn.ipynb) |
| 05 | [Variable stars](05_variable_stars.ipynb) | periods from combined ZTF + LSST light curves, aliases | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fforster/alerce-iucaa-2026/blob/main/05_variable_stars.ipynb) |

Each notebook starts with a single object and ends with a sample, and has exercises at the end. The three guiding
objects, all in the COSMOS field and seen by both surveys, are the RR Lyrae star `ZTF17aabhbva`, the type Ic-BL
supernova SN 2026gzf (`ZTF26aaonmha`) and the quasar `ZTF20acyxpwi`.

## Running

**Google Colab (recommended):** click a badge. The first cell installs what is needed and downloads `alerce_ms.py`.

**Locally:**

```bash
git clone https://github.com/fforster/alerce-iucaa-2026.git
cd alerce-iucaa-2026
pip install -r requirements.txt
jupyter lab
```

## Contents

* `alerce_ms.py` — a small, readable helper library (TAP queries, id conversion, look-up-table decoding,
  survey-independent light curves, features, probabilities, crossmatches, stamps, plotting). It caches every result
  on disk (`~/.cache/alerce_ms`) and retries politely when the servers are busy. Plots follow the ALeRCE conventions:
  band colours `BAND_COLORS` (u `#56B4E9`, g `#009E73`, r `#D55E00`, i `#E69F00`, z `#CC79A7`, y `#0072B2`), drawing
  order `BAND_ZORDER` (g on top, then r, i, z, u, y) and legends sorted by wavelength (`sort_legend`).
* `data/` — small precomputed tables (samples and DELIGHT host positions) so the sample-level sections run quickly
  and work even when the network is slow.
* `src/` — the notebooks in [jupytext](https://jupytext.readthedocs.io) percent format (edit these).
  `python tools/build.py [--kernel NAME] [names...]` converts them to `.ipynb` and executes them.

## Services used

| Service | URL |
|---|---|
| TAP (ADQL), schema `alerce_tap` | https://tap.alerce.online/tap |
| Multisurvey REST API | https://api-lsst.alerce.online |
| Experimental explorer (ALeRCE Labs) | https://explorer-experiment.alerce.online |
| catsHTM crossmatch | https://catshtm.alerce.online |
| LSST / ZTF explorers | https://lsst.alerce.online · https://alerce.online |

> **Please be gentle with the servers.** They are shared and rate-limited per IP address; a classroom behind one
> IP shares the same quota. Avoid loops of hundreds of requests — the notebooks load precomputed samples by default
> (`REBUILD = False`).

## Acknowledgements

Inspired by the [alercebroker/usecases](https://github.com/alercebroker/usecases) notebooks. The database
description follows the internal *ALeRCE book*. DELIGHT: Förster et al. 2022, AJ 164, 195.

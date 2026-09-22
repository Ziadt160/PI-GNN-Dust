# Riyadh 1 km PM Forecasting — Data Acquisition Spec

Goal: physics-informed **forecasting** of surface PM10/PM2.5 at **1 km** over a
regional box around Riyadh, using the validated PhysicsFluxLayer GNN backbone.

## 1. Domain / grid
- **Box (default):** lat **24.3–25.1 °N**, lon **46.3–47.1 °E** (~89 × 81 km).
- **Resolution:** 1 km → ~**89 × 81 ≈ 7,200 cells** (well within GNN range).
- Common grid that every dataset below is regridded onto (bilinear), exactly like
  `build_saudi_dataset.py` did for the coarse domain.

## 2. Datasets to download

| Role | Product | Native res | Source | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1 km spatial signal** | MODIS **MAIAC AOD** `MCD19A2` | 1 km | NASA LAADS DAAC / earthaccess | Tile **h22v06** covers the Riyadh box (verified against CMR 2026-09-20: 1,456 granules, ~18.2 GB for 2021-2024). The earlier note claiming h22v05/h23v05 was wrong -- v05 spans ~30-40N, Riyadh at 24.7N falls in v06. Daytime only; cloud gaps. |
| (optional, hourly) | Himawari / MSG **AOD** | ~few km | JAXA / EUMETSAT | Fills MODIS temporal gaps for forecasting. |
| **Point truth** | Ground-monitor PM10/PM2.5 | point | OpenAQ API / local PME network | For training + spatial cross-validation. |
| **Forecast forcing** (the future inputs) | ERA5 / ECMWF IFS **u10, v10, tp, blh, t2m, rh** | 0.1–0.25° | CDS API | Interpolated to 1 km as covariates; *future* values drive the rollout. |
| **Static 1 km covariates** | elevation (SRTM 30–90 m), land cover, NDVI, soil/erodibility, dist-to-source, population, roads | ≤1 km | USGS / Copernicus / OSM | Time-invariant node features. |

## 3. Target construction
- Train target = **surface PM from ground monitors** (the truth), with **MAIAC AOD as
  the 1 km spatial predictor** that carries the fine structure.
- A short **AOD→PM calibration** (regression vs monitors) gives a gap-filled 1 km PM
  *analysis* field used as the forecast **initial condition** and as a dense training
  target where monitors are absent.

## 4. Pipeline (to implement once data lands — mirrors build_saudi_dataset.py)
`build_riyadh_dataset.py`:
1. Read MAIAC tiles, monitors, ERA5 met, static rasters.
2. Regrid each onto the 1 km Riyadh grid (bilinear); convert any speed/dir winds to u/v.
3. Gap-handling for AOD (cloud mask) → flag/fill.
4. Build sequences `(C_t, u_t, v_t, covariates) -> C_{t+1}` and write `riyadh_merged.nc`.

`forecast_model.py` (already scaffolded + validated on synthetic data):
- PhysicsFluxLayer GNN, residual head, **multi-step rollout**.
- **CFL-stable** advection physics (see lesson below) for transport-consistent,
  non-blurring forecasts.

## 4b. Credentials & download status

| Dataset | Credential | Status |
| :--- | :--- | :--- |
| **ERA5** | `~/.cdsapirc` (configured) | ✅ **Working** — `download_era5_riyadh.py` (chunked monthly, auto unzip + merge + RH). |
| **MAIAC AOD** | Earthdata login → `~/.netrc` | ⛔ **You must set up.** Free account at https://urs.earthdata.nasa.gov, then `earthaccess.login(persist=True)` writes `~/.netrc`. |
| **Monitors** | OpenAQ API key | ⛔ **You must set up.** Free key at https://explore.openaq.org (account → API keys); export `OPENAQ_API_KEY`. |

> I can run downloads that use **already-configured** credentials, but I cannot create
> accounts or enter API keys/passwords. Set up the two ⛔ items above, then I'll write
> and run `download_maiac_riyadh.py` (earthaccess) and `download_monitors_riyadh.py`
> (OpenAQ API). **Note:** MAIAC over many years is large (~tens of GB even for 2 tiles)
> — scope the period to your training need.

ERA5 download commands:
```
python experiments/riyadh_1km/download_era5_riyadh.py test        # 1 month, validate
python experiments/riyadh_1km/download_era5_riyadh.py 2015 2024    # range, per-month files
```

## 5. Critical design constraint — CFL
At 1 km, a 5 m/s wind over a 1 h step moves **~18 cells** (Courant number ≫ 1). The
plain explicit-Euler advection residual is only valid for CFL < 1, so the 1 km
forecaster must use a **semi-Lagrangian or sub-stepped** advection term. The synthetic
validation in `forecast_model.py` demonstrates this directly (physics helps at CFL≈0.4,
hurts at CFL≈1.2).

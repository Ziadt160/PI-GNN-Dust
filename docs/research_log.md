# Research log — project state and how to resume

Written 2026-09-22 as the handover between working sessions, and kept as the
project's detailed lab notebook: every finding, the bug that would have invalidated
it, and what to try next. The [README](../README.md) has the summary; this is the
evidence behind it. §6c records the work committed after this log was written.

---

## 1. What this project now is

It **was**: a physics-informed GNN predicting CAMS reanalysis PM10 over Saudi Arabia.

It **is**: a rare-event classifier predicting **observed dust events** (airport
visibility) from ERA5 surface physics, keeping the graph transport model.

The pivot happened because independent validation showed the old target was broken.
That finding is the project's main result so far.

### Current honest claim

> Adds real, spatially-generalising skill at 24–48 h (1.9× persistence, 3.6×
> climatology, 1.45× gradient boosting). **Not usable as a warning system** —
> catching half of dust storms costs ~90% false alarms.

**Amended 2026-09-22 — the "2× below the WMO bar" line is withdrawn.** It compared
our `dust_storm` grade (vis < 1 km, base rate 1.26%) against published numbers
measured at 13–22%, and TS falls with the base rate whether or not the forecast
got worse. Measured in `tests/test_metrics.py`: at *constant* forecast
quality TS drops 11.4× from our `dusty` grade to our `severe` grade while SEDI
does not move at all. Compare at **`dusty`** (vis < 5 km, base rate **8.09%**),
which is the band every published dust-event definition sits in, or compare with
**SEDI**, which is base-rate independent. And the 0.31 itself was never traced to
a primary source — see the `WMO SDS-WAS` entry in `src/metrics/literature.py`.

Never quote a CAMS-scored number as a forecast skill number. Report both:
`TS 0.77 vs CAMS reanalysis` and `TS 0.14 vs station observations` are different
measurements and only the second means anything.

---

## 2. Findings, all measured

| # | Finding | Evidence | Script |
|---|---|---|---|
| 1 | **CAMS PM10 under-represents dust storms by 15–49×** | 3,369 dust-coded storm hours: CAMS median 268 µg/m³ vs 3,912–13,040 required by Koschmieder. Never reached the lower bound once. **Fog control: 48 µg/m³** — correctly clean, so the field is not uniformly biased, it specifically fails on dust | ad-hoc, see §7 |
| 2 | **Model rankings invert between targets** | Best-vs-CAMS model is 2nd vs observations; the *unmodified baseline* is 6th vs CAMS, **1st** vs real storms | `validate_vs_metar.py` |
| 3 | **42% of dust storms are imported** | Local `u*` below threshold on 1,409 of 3,369 storm hours; 68% arrive on a northerly wind | ad-hoc |
| 4 | **Upwind stations lead by up to a day** | Interior storm preceded by a *northern* station's storm **48.8%** within 24 h vs **21.8%** from the south — directional, not widespread | ad-hoc |
| 5 | **The model is mostly a wind forecast** | future met 40% + emission 18% ≈ 58%; CAMS PM10 **5%**; graph transport **3%** | `feature_importance.py` |
| 6 | **No usable operating point** | catch 50% → FAR 0.90, 18 false alarms/station/yr; catch 80% → FAR 0.95 | `operating_points.py` |
| 7 | **SoilGrids is useless over Arabia** | 0/20 cells above 80% sand, total spread 22 pp, in a country of sand seas | `download_soilgrids.py` |
| 8 | **Measured ERA5 drivers changed nothing** | 6 horizon/threshold pairs all within ±0.025 noise, despite emission-active fraction 5.1% → 19.5% | two `station_daily_interior*.json` |
| 9 | **AI-DUST's gated physics loss is inert** | Eq.1 forces the Eq.3 ratio into (0,2) so the `<10` test always passes; measured activation 0.1–0.3% | `gated_loss.py` docstring |
| 10 | **AI-DUST's Eq.1 reference is CFL-unstable** | Courant ≈3.9 at their own 0.25°/3h | `transport.py` |
| 11 | **AI-DUST's 15×15 patch is oversized** | 3 GAT layers give a Manhattan radius-3 field; 7×7 is bit-identical at 4.6× less compute | `registry.py` |

### Best model vs all controls — 24 h, daily dust-days, station-verified

| | TS | ETS | POD | FAR |
|---|---:|---:|---:|---:|
| never-dust | 0.000 | 0.000 | 0.000 | — |
| climatology (station × month) | 0.039 | 0.033 | 0.092 | 0.937 |
| wind threshold | 0.072 | 0.067 | 0.135 | 0.865 |
| logistic regression | 0.072 | 0.067 | 0.135 | 0.865 |
| persistence | 0.076 | 0.070 | 0.141 | 0.859 |
| gradient boosting | 0.098 | 0.093 | 0.178 | 0.822 |
| **flux-GNN** | **0.142** | **0.137** | **0.249** | **0.751** |
| *WMO SDS-WAS operational* | *≈0.31* | | | |

`results/baselines_station.json`. This is the table to update and re-quote.

---

## 3. Where I was wrong (don't re-derive these)

- **"ETS will shrink our numbers."** It doesn't — at a 1% base rate chance hits are
  ~9% of actual hits, so ETS ≈ TS. ETS corrects for *luck*; the inflation here is
  *persistence*. Use `add_skill_scores` (skill vs persistence) instead.
- **"The physics anchor is the valuable AI-DUST transplant."** Backwards: bounded
  head +0.037 TS, anchor +0.003 with *worse* RMSE.
- **"Stacking future-met and emission is anti-additive."** That was the seeding bug.
- **"4% of cell-times exceed CFL 1."** Full record: mean 0.14, **0.4%**.
- **"Shamals are a few km across."** They are synoptic — hundreds of km. This is why
  resolution is *not* the bottleneck.

---

## 4. Bugs fixed — all silent, all would have invalidated results

| Bug | Symptom |
|---|---|
| Seeding once per process | sd 0.033 TS between identical configs — larger than most effects measured |
| MAIAC double-scaling | netCDF4 auto-applies `scale_factor`; extra ×0.001 made AOD 1000× too small → all-zero erodibility map |
| Pixel-vs-observation counting | 740,000 "observations" per cell; conditional probability became a pixel exceedance fraction |
| Frequency-matching a constant score | "never dust" silently became "always dust", POD 1.0 |
| Date-only daily aggregation | collapsed 16 stations into one national series, base rate 1.03% → 7.47% |
| **Hops not scaled to horizon** | 2 hops = 152 km reach while 24 h transport is 501 km |
| Boundary stations dropped as *inputs* | correct to exclude as targets, wrong as inputs — discarded the 48.8% upwind lead |
| `_wetness_proxy` shape | assumed `[T,N]`, received `[T,H,W]` |
| Zero training labels | METAR covered 2021–24 while training on 2003–18; loss was exactly 0.0000 and looked plausible |

Guards now in place: per-model seeding, driver-coverage threshold (98%), zero-label
`RuntimeError`, constant-score guard, provenance printing on every run.

---

## 5. Code — 57 modules, ~8,700 lines

### Physics
- `src/physics/emission.py` — Shao saltation, Fécan moisture, MB95 drag partition
  (**off by default**: ERA5 `fsr` is aerodynamic not aeolian, zeroes 99% of cells),
  ERA5-or-proxy with `provenance()`, **erodibility map hook**
- `src/physics/transport.py` — `∇·(VC)`, `u·∇C`, Euler + **semi-Lagrangian** `C_ref`,
  Courant diagnostic. Sign derived from the coordinate vector
- `src/physics/gated_loss.py` — AI-DUST Eq.2+3 with activation logging

### Models
- `src/models/dust_transport.py` — **`DustTransportGNN`, the current best**: every
  node emits and advects, emission re-injected each hop, `hops_for_horizon()`,
  readout at core nodes only
- `src/models/heads.py` / `backbones.py` / `unified.py` / `registry.py` — the 29-model
  AI-DUST comparison framework (largely superseded; keep for the write-up)

### Data
- `src/data/saudi_dataset.py` — `SaudiFields`, node-level cache, chunked derivatives
- `src/data/topologies.py` — feature assembly (`met_offset`!), full + patch datasets
- `src/data/station_target.py` — METAR visibility → dust labels, Koschmieder, censoring
- `src/data/rollout.py`, `patch_frames.py`

### Metrics
- `src/metrics/aidust_metrics.py` — TS, **ETS, HSS**, POD, FAR, **FGE**, multi-threshold,
  rollout, skill-vs-persistence
- `src/metrics/literature.py` — 8 published references with `comparable: False` flags

### Experiments
`experiments/station_dust/`: `train_station_dust.py` (main), `baselines_station.py`,
`feature_importance.py`, `operating_points.py`.
`experiments/saudi_pm10/`: `validate_vs_metar.py`, `rescore.py`.
`experiments/data_prep/`: `build_erodibility_map.py`, `download_*.py`.
The physics audit is `tests/test_dust_physics.py` (**re-run after any change**:
`pytest tests/`).

---

## 6. Data

| Asset | State |
|---|---|
| CAMS Saudi 0.75° 22 yr | ✅ 445 MB |
| ERA5 emission drivers 22/22 yr | ✅ 367 MB |
| MAIAC AOD 1,210 granules | ✅ 14 GB |
| METAR Saudi 26 stations | ✅ 142 MB, 3.56 M obs |
| **METAR WIDE 224 stations** | ✅ **941 MB, 23.6 M obs**, 24 upwind (Iraq 14, Syria 5, Jordan 3, Kuwait 2) |
| Erodibility map | ✅ 28 KB, spread 0.286 |
| **CAMS wide** | ⏳ **20/22** (missing 2023, 2024) |
| **ERA5 wide** | ⏳ **2 files only** — barely started |

Credentials the downloaders expect: `~/.cdsapirc` (ECMWF CDS **and** ADS, with the
dataset licences accepted), `~/.edl_token` (NASA Earthdata, for MAIAC),
`~/.openaq_key` (OpenAQ).

The large datasets are found under `DUST_DATA_ROOT` (default: the repo's `data/`;
see `src/paths.py`). Results were produced with Python 3.10, torch 2.5.1+cu121 on a
4 GB GTX 1650; versions are pinned in `requirements.txt`.

---

## 6b. Added 2026-09-22 — literature review, SEDI, levels, ordinal head

Commits `b8d4d65` and `4693333`.

| What | State |
|---|---|
| **SEDI / EDI** in `aidust_metrics`, base rates on every scored table, raw preds saved to `results/preds/` | ✅ done and measured |
| **`tests/test_metrics.py`** — 19 checks: base-rate dependence, SEDI properties, ordinal head | ✅ 19/19 |
| **`src/physics/stability.py`** — t_500, t_700-500, Ri, shear, low-level jet | ✅ built, 6 new checks in `tests/test_dust_physics.py` (now 14/14) |
| **`download_era5_levels.py`** — 440 parallel CDS requests, ~2.6 GB | ⏳ **running**, hours to go |
| **`src/models/ordinal.py`** — SORD head, monotone tail sums | ✅ built and verified |
| **`ablation_levels.py`** — base / +levels / +ordinal / both, multi-seed | ⏳ base vs ordinal running; levels arm blocked on the download |

Measured base rates (daily, interior, test 2021–2024) — quote these, not estimates:
`dusty` **8.09%**, `dust_storm` **1.26%**, `severe` **0.23%**.

Two things that bit, both now regression-tested:

- **Inverse-frequency class weights normalised to mean 1** drove the majority
  grade to 0.004, so the model would have got almost no gradient for saying
  "clear". Weights are now relative to the most common grade.
- **ERA5 returns `pressure_level` ascending by height** (900 first), not in the
  order the level list is written. Every access goes through a value lookup;
  do not replace it with positional indexing.

The CDS **pressure-level** cost limit is far tighter than single-levels: 1 variable
× 1 quarter is accepted, 5 variables × 1 month is refused. Nearly all the ~8 min
per request is queue wait, hence the thread pool.

## 6c. Added after this log — physics encodings and the normalisation bug

Commits `4331a4a` to `165c45f`, summarised from their messages.

- **Missing physics encoded as features and architecture, not as loss penalties**
  (`--physics-features`, `--deposition`, `--conservative`): hour of day, wet and dry
  removal as a per-node survival fraction, a threshold-excess channel, soil-moisture
  memory, and a mass-conserving flux layer. The loss route had already been measured
  as inert (§2, finding 9).
- **All three together collapsed the model** to a near-constant predictor. An
  isolation queue (`experiments/station_dust/run_queue.py`) traced it to the
  trainer, not the physics: normalisation statistics were taken every 584th
  sample, an exact multiple of 8 steps per day, so every sample landed at midnight,
  hour-of-day got std 0, and the 1e-6 floor turned it into a 1e6 amplifier. Fixed
  in `cbcc2e2` (random subsample; near-constant channels get std 1).
- **Labels are now gated on the METAR weather code** (`--dust-gate`): 36% of
  `severe` visibility events were fog or rain, not dust. The gate screens labels
  only; the observation still feeds the model as input.
- **Not yet measured:** whether the physics encodings help once the inputs are
  sane (`ctrl_fix` vs `phys_fix`), and the dense CAMS auxiliary head
  (`--aux-cams`). Both are queued in `run_queue.py`; their outputs go to the
  untracked `results/ablation/`.

## 7. Resume here

From the repository root, with the data located (see `src/paths.py`):

```bash
export DUST_DATA_ROOT="/path/to/DustProject/data"
```

**1. Finish the downloads** (ERA5 is ~12 h for 88 quarterly requests; consider
narrowing to 2015–2024 to halve it):
```bash
python -u experiments/data_prep/download_wide_domain.py cams 2023 2024
python -u experiments/data_prep/download_wide_domain.py era5 2003 2024
```

**2. Re-run the interrupted training** (never produced output):
```bash
python -u experiments/station_dust/train_station_dust.py --epochs 60 --obs-input --upwind-obs \
    --interior-only --daily --transport --hidden 64 --buffer-rings 2 \
    --out results/full_buffer2.json
```
Cost: 3 h ≈ 10 min, 24 h ≈ 38 min, 48 h ≈ 60 min. Buffer 3 was requested by the
user (concentric rings: outer 3 rings input, core predicted) but leaves only
**4 stations** on this domain — it is the right design waiting for the wide domain.

**3. Then**: build `wide_merged.nc` (all land unmasked, not just Saudi), retrain,
and **re-run `baselines_station.py`**. The headline test is whether the
**GNN-vs-gradient-boosting margin widens** from 0.142 vs 0.098. If it does, the
graph transport is finally working now that the source region exists.

---

## 8. Open questions

1. **Does the wide domain help?** 42% of storms are imported and currently
   unpredictable in principle. This is the biggest open lever.
2. **Does the erodibility map help?** Built and wired but never trained with.
   Caveat: 122 MAIAC days, MODIS overpass ~10:30 local — biased to daytime sources.
3. **Full leave-one-station-out.** `--loso` exists, never run. Withholding the
   observation costs only 0.001 at 24 h, which is strong but not the same test.
4. **`speed` (current) has negative importance** (−0.0124, permuting it *helps*).
   Drop it; likely a free gain.
5. **Contact the WMO SDS-WAS GCC node** (Jeddah, Saudi NCM, `sdsc@ncm.gov.sa`) —
   the operational system for this exact domain, launched 2023. They hold the
   PME/NCM station network and are the natural comparison and collaborator.

## 9. What NOT to do

- **Don't run DeepCAMS / 1 km downscaling yet.** It sharpens CAMS, which the model
  already ignores (5% importance) and which is 15–49× too low. DeepCAMS's own
  validation moved R vs stations by 0.004. Fix the target first.
- **Don't re-run the 29-model AI-DUST ablation.** Those rankings don't transfer off
  the CAMS target. `baselines_station.py` is the informative comparison.
- **Don't trust SoilGrids** over this domain.

---

## 10. Deliverable

Supervisor report, v5:
[`dust_storm_forecasting_report_2026-09-22.pdf`](dust_storm_forecasting_report_2026-09-22.pdf).
Numbers in it are current as of the 15–49× correction; the buffer-ring and
wide-domain work is not in it yet, nor is anything in §6c.

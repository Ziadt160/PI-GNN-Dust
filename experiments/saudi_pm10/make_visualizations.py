"""Generate comparison charts (verified numbers) + Saudi dust maps & a dust-transport
animation (real CAMS PM10/PM2.5 + ERA5 winds). Outputs to results/figures/.

Run:  python -u experiments/saudi_pm10/make_visualizations.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
from src import paths  # noqa: E402

MERGED = paths.SAUDI_MERGED
DUST_CMAP = "YlOrBr"

# ---------------- Verified results (Saudi test 2021-24, PM10) ----------------
# from experiments/saudi_pm10/physics_metrics.py + physics_compare.py + diagnostics.py
MODELS = ["Persistence", "Ridge", "RandomForest", "GradBoost",
          "PI-GNN\ncomplete", "PI-GNN\nadvection", "PI-GNN\nnone"]
FAMILY = ["baseline", "classical", "classical", "classical", "pi-gnn", "pi-gnn", "pi-gnn"]
R2 = np.array([0.332, 0.496, 0.516, 0.534, 0.127, 0.471, 0.616])
ADV_RESID = np.array([0.399, 0.518, 0.508, 0.516, 0.249, 0.254, 0.541])
MASS_ERR = np.array([0.0, 10.07, 9.96, 9.50, 14.88, 7.84, 5.00])
ROUGH = np.array([1.00, 0.50, 0.51, 0.52, 2.33, 1.00, 0.60])
GT_ADV = 0.679
COL = {"baseline": "#888888", "classical": "#4C72B0", "pi-gnn": "#C44E52"}
colors = [COL[f] for f in FAMILY]


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}", flush=True)


# ============================ COMPARISON CHARTS ============================
def chart_accuracy():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = ax.bar(range(len(MODELS)), R2, color=colors, edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_ylabel("R²  (test 2021–2024, PM10)"); ax.set_ylim(0, 0.7)
    ax.set_title("Forecast accuracy — PI-GNN vs classical baselines (Saudi domain)")
    for r, v in zip(b, R2):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(R2[0], color="#888", ls="--", lw=1)
    ax.text(len(MODELS) - 0.4, R2[0] + 0.005, "persistence", color="#888", fontsize=7, ha="right")
    save(fig, "01_accuracy_r2.png")


def chart_skill():
    skill = (1 - (1 - R2) / (1 - R2[0])) * 100  # MSE reduction vs persistence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = ax.bar(range(len(MODELS)), skill, color=colors, edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_ylabel("Skill vs persistence (% MSE reduction)")
    ax.set_title("Forecast skill over persistence baseline (Saudi domain)")
    for r, v in zip(b, skill):
        ax.text(r.get_x() + r.get_width() / 2, v + (0.8 if v >= 0 else -2.2),
                f"{v:+.0f}%", ha="center", fontsize=8)
    save(fig, "02_skill_vs_persistence.png")


def chart_physics_metrics():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [(ADV_RESID, "Advection residual ↓\n(transport consistency)", GT_ADV, "ground truth"),
             (MASS_ERR, "Mass-conservation error % ↓", 0, None),
             (ROUGH, "Spatial roughness / ground truth\n(1.0 = realistic, <1 blurry)", 1.0, "ideal")]
    for ax, (vals, title, ref, reflab) in zip(axs, specs):
        b = ax.bar(range(len(MODELS)), vals, color=colors, edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m.replace("\n", " ") for m in MODELS], fontsize=6.5, rotation=40, ha="right")
        ax.set_title(title, fontsize=9)
        if reflab:
            ax.axhline(ref, color="green", ls="--", lw=1.2)
            ax.text(0.02, ref, reflab, color="green", fontsize=7, va="bottom", transform=ax.get_yaxis_transform())
    fig.suptitle("Physical-fidelity metrics — accurate models are physically worse", fontsize=11)
    save(fig, "03_physics_metrics.png")


def chart_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, m in enumerate(MODELS):
        ax.scatter(R2[i], ROUGH[i], s=170, color=colors[i], edgecolor="k", zorder=3)
        ax.annotate(m.replace("\n", " "), (R2[i], ROUGH[i]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(1.0, color="green", ls="--", lw=1.3)
    ax.text(0.34, 1.03, "realistic spatial structure", color="green", fontsize=8)
    ax.axhspan(0, 0.7, color="orange", alpha=0.06)
    ax.text(0.6, 0.45, "blurry / over-smoothed", color="darkorange", fontsize=9, ha="center")
    ax.set_xlabel("Accuracy  →  R² (test 2021–2024)")
    ax.set_ylabel("Spatial fidelity  →  roughness / ground truth")
    ax.set_title("The accuracy ↔ physical-fidelity trade-off\n"
                 "classical/data-driven win R² but blur the field; physics keeps it realistic")
    ax.set_ylim(0, 2.5); ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[k], label=k) for k in COL], loc="upper left", fontsize=8)
    save(fig, "04_accuracy_vs_fidelity_tradeoff.png")


def chart_resolution():
    # old single-box diagnostics: physics on/off at 12h vs 6h
    groups = ["12 h horizon", "6 h horizon"]
    persist = [0.527, 0.639]; off = [0.678, 0.773]; on = [0.573, 0.713]
    x = np.arange(2); w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w, persist, w, label="Persistence", color="#888")
    ax.bar(x, off, w, label="PI-GNN (data-driven)", color="#C44E52")
    ax.bar(x + w, on, w, label="PI-GNN (physics loss)", color="#DD8452")
    for i in range(2):
        for xx, vv in [(x[i]-w, persist[i]), (x[i], off[i]), (x[i]+w, on[i])]:
            ax.text(xx, vv + 0.008, f"{vv:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("R²"); ax.set_ylim(0, 0.85)
    ax.set_title("Resolution & the physics-loss effect (single-box diagnostics)")
    ax.legend(fontsize=8)
    save(fig, "05_resolution_physics.png")


# ============================ DUST MAPS ============================
def _basemap(ax, lon, lat):
    ax.set_extent([float(lon.min()) - 0.4, float(lon.max()) + 0.4,
                   float(lat.min()) - 0.4, float(lat.max()) + 0.4], ccrs.PlateCarree())
    try:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f5f0", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, ls=":")
    except Exception as e:
        print(f"  (map features unavailable: {e})")
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}


def load_data():
    ds = xr.open_dataset(MERGED)
    lat = ds.latitude.values; lon = ds.longitude.values
    mask = ds.valid_mask.values
    pm10 = ds["pm10"].values * 1e9  # ug/m3
    pm25 = ds["pm2p5"].values * 1e9
    u = ds["u10"].values; v = ds["v10"].values
    t = ds.valid_time.values
    pm10 = np.where(mask, pm10, np.nan)
    pm25 = np.where(mask, pm25, np.nan)
    # domain-mean PM10 over valid cells, to locate a dust event
    dm = np.nanmean(pm10.reshape(pm10.shape[0], -1), axis=1)
    return lat, lon, mask, pm10, pm25, u, v, t, dm


def map_snapshot(lat, lon, pm10, u, v, t, idx, vmax):
    LON, LAT = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    _basemap(ax, lon, lat)
    pc = ax.pcolormesh(LON, LAT, pm10[idx], cmap=DUST_CMAP, vmin=0, vmax=vmax,
                       shading="nearest", transform=ccrs.PlateCarree(), zorder=2, alpha=0.85)
    step = 1
    ax.quiver(LON[::step, ::step], LAT[::step, ::step], u[idx][::step, ::step], v[idx][::step, ::step],
              transform=ccrs.PlateCarree(), scale=200, width=0.0025, color="black", alpha=0.6, zorder=3)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02); cb.set_label("PM10  (µg/m³)")
    ax.set_title(f"Saudi dust field + 10 m wind — {str(t[idx])[:13]}", fontsize=11)
    save(fig, "06_dust_map_snapshot.png")


def map_sequence(lat, lon, pm10, u, v, t, start, vmax, n=6, stride=2):
    LON, LAT = np.meshgrid(lon, lat)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    for k, ax in enumerate(axs.ravel()):
        i = start + k * stride
        _basemap(ax, lon, lat)
        pc = ax.pcolormesh(LON, LAT, pm10[i], cmap=DUST_CMAP, vmin=0, vmax=vmax,
                           shading="nearest", transform=ccrs.PlateCarree(), zorder=2, alpha=0.85)
        ax.quiver(LON, LAT, u[i], v[i], transform=ccrs.PlateCarree(),
                  scale=220, width=0.003, color="black", alpha=0.55, zorder=3)
        ax.set_title(str(t[i])[:13], fontsize=9)
    fig.suptitle("Dust-transport simulation — PM10 advecting with the wind (CAMS/ERA5 reanalysis)", fontsize=12)
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(pc, cax=cax).set_label("PM10 (µg/m³)")
    save(fig, "07_dust_simulation_sequence.png")


def map_pm10_vs_pm25(lat, lon, pm10, pm25, t, idx):
    LON, LAT = np.meshgrid(lon, lat)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, field, name, vmx in [(axs[0], pm10[idx], "PM10", np.nanmax(pm10[idx])),
                                 (axs[1], pm25[idx], "PM2.5", np.nanmax(pm25[idx]))]:
        _basemap(ax, lon, lat)
        pc = ax.pcolormesh(LON, LAT, field, cmap=DUST_CMAP, vmin=0, vmax=vmx,
                           shading="nearest", transform=ccrs.PlateCarree(), zorder=2, alpha=0.85)
        fig.colorbar(pc, ax=ax, shrink=0.75).set_label(f"{name} (µg/m³)")
        ax.set_title(name, fontsize=11)
    fig.suptitle(f"Multi-task targets: PM10 vs PM2.5 — {str(t[idx])[:13]}", fontsize=12)
    save(fig, "08_pm10_vs_pm25.png")


def map_event_vs_clean(lat, lon, pm10, t, hi, lo, vmax):
    LON, LAT = np.meshgrid(lon, lat)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, idx, lab in [(axs[0], hi, "Dust event (high PM10)"), (axs[1], lo, "Clean day (low PM10)")]:
        _basemap(ax, lon, lat)
        pc = ax.pcolormesh(LON, LAT, pm10[idx], cmap=DUST_CMAP, vmin=0, vmax=vmax,
                           shading="nearest", transform=ccrs.PlateCarree(), zorder=2, alpha=0.85)
        fig.colorbar(pc, ax=ax, shrink=0.75).set_label("PM10 (µg/m³)")
        ax.set_title(f"{lab}\n{str(t[idx])[:13]}", fontsize=10)
    fig.suptitle("Dust event vs clean day over Saudi Arabia", fontsize=12)
    save(fig, "09_event_vs_clean.png")


def animate(lat, lon, pm10, u, v, t, start, vmax, frames=20, stride=1):
    LON, LAT = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(8, 6.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    _basemap(ax, lon, lat)
    pc = ax.pcolormesh(LON, LAT, pm10[start], cmap=DUST_CMAP, vmin=0, vmax=vmax,
                       shading="nearest", transform=ccrs.PlateCarree(), zorder=2, alpha=0.85)
    qv = ax.quiver(LON, LAT, u[start], v[start], transform=ccrs.PlateCarree(),
                   scale=220, width=0.003, color="black", alpha=0.55, zorder=3)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8); cb.set_label("PM10 (µg/m³)")
    title = ax.set_title("")

    def upd(f):
        i = start + f * stride
        pc.set_array(pm10[i][:-1, :-1].ravel() if False else pm10[i].ravel())
        qv.set_UVC(u[i], v[i])
        title.set_text(f"Saudi dust-transport simulation — {str(t[i])[:13]}")
        return pc, qv, title

    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    p = os.path.join(OUT, "10_dust_simulation.gif")
    anim.save(p, writer=animation.PillowWriter(fps=4))
    plt.close(fig)
    print(f"  saved 10_dust_simulation.gif", flush=True)


def main():
    print("Comparison charts ...", flush=True)
    chart_accuracy(); chart_skill(); chart_physics_metrics(); chart_tradeoff(); chart_resolution()

    print("Loading Saudi fields for dust maps ...", flush=True)
    lat, lon, mask, pm10, pm25, u, v, t, dm = load_data()
    event = int(np.argmax(dm))                       # peak dust timestep
    clean = int(np.argmin(dm[dm > 0]) if (dm > 0).any() else 0)
    vmax = float(np.nanpercentile(pm10[event], 98))
    print(f"  dust event at {str(t[event])[:13]} (mean {dm[event]:.0f} ug/m3); vmax~{vmax:.0f}", flush=True)
    start = max(0, event - 6)
    print("Dust maps ...", flush=True)
    map_snapshot(lat, lon, pm10, u, v, t, event, vmax)
    map_sequence(lat, lon, pm10, u, v, t, start, vmax)
    map_pm10_vs_pm25(lat, lon, pm10, pm25, t, event)
    map_event_vs_clean(lat, lon, pm10, t, event, clean, vmax)
    print("Animation ...", flush=True)
    animate(lat, lon, pm10, u, v, t, start, vmax, frames=20, stride=1)
    print(f"\nAll figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()

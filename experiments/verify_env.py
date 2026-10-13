"""Check that the environment and data are ready.

The committed Gulf sample (data/*.nc) is required. The large Saudi datasets are
reported but optional: only experiments/saudi_pm10 and experiments/station_dust
need them, and they are looked up under DUST_DATA_ROOT (see src/paths.py).

Run from the repository root:  python experiments/verify_env.py
"""
import os
import sys
import importlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from src import paths  # noqa: E402

REQUIRED = ["numpy", "scipy", "pandas", "torch", "torch_geometric", "xarray",
            "netCDF4", "sklearn", "matplotlib", "yaml", "tqdm"]


def main():
    print("=== Environment check ===")
    ok = True
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [ok] {mod:<16} {ver}")
        except Exception as exc:
            ok = False
            print(f"  [MISSING] {mod:<16} ({exc})")

    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except Exception:
        pass

    print("\n=== Gulf sample (committed, required) ===")
    data_dir = os.path.join(REPO_ROOT, "data")
    expected = ["data_sfc.nc"] + [f"era5_{y}.nc" for y in (2003, 2004, 2005, 2006, 2007)]
    for fname in expected:
        path = os.path.join(data_dir, fname)
        status = "ok" if os.path.exists(path) else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  [{status}] {fname}")

    print(f"\n=== Saudi datasets (optional) under {paths.DATA_ROOT} ===")
    for label, path in [("merged CAMS + ERA5", paths.SAUDI_MERGED),
                        ("ERA5 emission drivers", paths.EMISSION_DIR),
                        ("METAR stations", paths.METAR_DIR),
                        ("ERA5 pressure levels", paths.LEVELS_DIR)]:
        status = "ok" if os.path.exists(path) else "absent"
        print(f"  [{status}] {label:<24} {os.path.relpath(path, paths.DATA_ROOT)}")

    print("\n" + ("All required checks passed." if ok else "Some checks FAILED - see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

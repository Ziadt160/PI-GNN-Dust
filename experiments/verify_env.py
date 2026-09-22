"""Validate that the runtime environment and data are ready for the PI-GNN.

Run from the repository root:  python experiments/verify_env.py
"""
import os
import sys
import importlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED = ["numpy", "torch", "torch_geometric", "xarray", "netCDF4",
            "sklearn", "matplotlib", "yaml", "tqdm"]


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

    print("\n=== Data check ===")
    data_dir = os.path.join(REPO_ROOT, "data")
    expected = ["data_sfc.nc"] + [f"era5_{y}.nc" for y in (2003, 2004, 2005, 2006, 2007)]
    for fname in expected:
        path = os.path.join(data_dir, fname)
        status = "ok" if os.path.exists(path) else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  [{status}] {fname}")

    print("\n" + ("All checks passed." if ok else "Some checks FAILED - see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

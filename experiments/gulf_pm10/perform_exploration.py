"""Summarise the Gulf PM10 field and plot its domain-mean time series.

Writes ``results/exploration_summary.txt`` and ``results/pm10_timeseries.png``.

Run from the repository root:  python experiments/gulf_pm10/perform_exploration.py
"""
import os

import matplotlib.pyplot as plt
import xarray as xr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SFC = os.path.join(REPO_ROOT, "data", "data_sfc.nc")
RESULTS = os.path.join(REPO_ROOT, "results")


def main():
    ds = xr.open_dataset(SFC)
    pm10 = ds['pm10']

    with open(os.path.join(RESULTS, 'exploration_summary.txt'), 'w') as f:
        f.write("Data Exploration Summary\n")
        f.write("========================\n")
        f.write(f"Dataset: {ds}\n\n")
        f.write("Variable: pm10\n")
        f.write(f"Mean: {pm10.mean().values:.4e} {pm10.units}\n")
        f.write(f"Max:  {pm10.max().values:.4e} {pm10.units}\n")
        f.write(f"Min:  {pm10.min().values:.4e} {pm10.units}\n")

    plt.figure(figsize=(10, 6))
    pm10.mean(dim=['latitude', 'longitude']).plot()
    plt.title('Spatially Averaged PM10 over Time')
    plt.ylabel(f'PM10 ({pm10.units})')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS, 'pm10_timeseries.png'))
    print("Wrote results/exploration_summary.txt and results/pm10_timeseries.png")


if __name__ == "__main__":
    main()

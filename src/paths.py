"""Where the large, uncommitted datasets live.

Raw and merged inputs (CAMS, ERA5, METAR, MAIAC; ~17 GB) are not in the
repository. They are looked up under ``DUST_DATA_ROOT``, which defaults to the
repository's own ``data/`` folder. Point it elsewhere when the data lives on
another disk, or when running from a git worktree, which has no ``data/`` of
its own:

    export DUST_DATA_ROOT="/path/to/DustProject/data"          # bash
    $env:DUST_DATA_ROOT = "D:/path/to/DustProject/data"         # PowerShell

The small 2003-2007 Gulf sample used by ``experiments/gulf_pm10`` is committed
under ``data/`` and is always read from the repository itself.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("DUST_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

SAUDI_DIR = os.path.join(DATA_ROOT, "ALL DATA SAUDI")
WIDE_DIR = os.path.join(DATA_ROOT, "ALL DATA WIDE")
RIYADH_DIR = os.path.join(DATA_ROOT, "ALL DATA RIYADH")

SAUDI_MERGED = os.path.join(SAUDI_DIR, "saudi_merged.nc")
METAR_DIR = os.path.join(SAUDI_DIR, "metar")
EMISSION_DIR = os.path.join(SAUDI_DIR, "emission")
LEVELS_DIR = os.path.join(SAUDI_DIR, "levels")
MAIAC_DIR = os.path.join(SAUDI_DIR, "maiac")
ERODIBILITY = os.path.join(SAUDI_DIR, "erodibility.nc")
SOILGRIDS = os.path.join(SAUDI_DIR, "soilgrids_texture.nc")

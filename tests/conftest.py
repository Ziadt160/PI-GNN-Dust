"""Put the repository root on sys.path, so tests import ``src`` and ``experiments``
the same way the scripts do, without installing anything."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

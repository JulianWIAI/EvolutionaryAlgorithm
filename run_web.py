"""
run_web.py — Smart Swarm web launcher.

Use this instead of 'python -m pygbag .' to start the browser version.

    python run_web.py

What it does
------------
Pygbag 0.9.3 expects browserfs.min.js from its CDN, but that file no longer
exists there (HTTP 404).  Without BrowserFS, Python cannot mount the virtual
filesystem inside the browser and cannot import any project modules.

This script pre-populates Pygbag's local cache with BrowserFS 1.4.3 fetched
from jsDelivr before handing control to Pygbag.  Pygbag checks the cache
before attempting a remote download, so it finds the file immediately and
never hits the broken CDN URL.

The cache is persistent between runs, so the download only happens once.
"""

import hashlib
import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# BrowserFS cache setup
# ---------------------------------------------------------------------------

# Pygbag caches remote files using the MD5 of the URL as the filename.
# This is the MD5 of the exact (double-slash) URL Pygbag 0.9.3 requests:
#   https://pygame-web.github.io/cdn/0.9.3//browserfs.min.js
_BFS_CACHE_KEY = "36fb7f27c808bd4880632a1d15a4be0c"
_BFS_CDN_URL   = "https://cdn.jsdelivr.net/npm/browserfs@1.4.3/dist/browserfs.min.js"

_CACHE_DIR   = os.path.join(os.path.dirname(__file__), "build", "web-cache")
_DATA_FILE   = os.path.join(_CACHE_DIR, f"{_BFS_CACHE_KEY}.data")
_HEAD_FILE   = os.path.join(_CACHE_DIR, f"{_BFS_CACHE_KEY}.head")

# Pygbag's testserver reads a .head file alongside every .data file.
# It parses "Key: Value" lines and sends them as HTTP response headers,
# stopping at the first line that does not contain ": ".
_HEAD_CONTENT = "Content-Type: application/javascript\n"


def _ensure_browserfs():
    """Download BrowserFS into Pygbag's cache if not already present.

    Pygbag's testserver.py requires two files per cached resource:
      <hash>.data  — the file content
      <hash>.head  — HTTP response headers (newline-separated key: value pairs)
    Both must exist or the server raises FileNotFoundError when serving.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    # --- .data file (the actual BrowserFS JavaScript) -----------------------
    if os.path.isfile(_DATA_FILE) and os.path.getsize(_DATA_FILE) > 0:
        print(f"[run_web] BrowserFS already cached ({os.path.getsize(_DATA_FILE):,} bytes).")
    else:
        print("[run_web] Downloading BrowserFS 1.4.3 from jsDelivr ...")
        try:
            urllib.request.urlretrieve(_BFS_CDN_URL, _DATA_FILE)
            print(f"[run_web] Done — {os.path.getsize(_DATA_FILE):,} bytes cached.")
        except Exception as exc:
            print(f"[run_web] WARNING: Could not download BrowserFS: {exc}")
            print("[run_web] The simulation may show a grey screen in the browser.")
            return

    # --- .head file (HTTP headers Pygbag's server reads before serving) -----
    if not os.path.isfile(_HEAD_FILE):
        with open(_HEAD_FILE, "w") as fh:
            fh.write(_HEAD_CONTENT)
        print("[run_web] Created BrowserFS header file.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ensure_browserfs()

    # Hand off to Pygbag — replaces this process so the terminal output is clean.
    print("[run_web] Starting Pygbag — open http://localhost:8000\n")
    os.execv(sys.executable, [sys.executable, "-m", "pygbag", "."])

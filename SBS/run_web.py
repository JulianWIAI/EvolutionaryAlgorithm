"""
run_web.py — Smart Swarm web launcher.

Use this instead of 'python -m pygbag .' to start the browser version.

    python SBS/run_web.py

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

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

# ---------------------------------------------------------------------------
# BrowserFS cache setup
# ---------------------------------------------------------------------------

# Pygbag caches remote files using the MD5 of the URL as the filename.
# This is the MD5 of the exact (double-slash) URL Pygbag 0.9.3 requests:
#   https://pygame-web.github.io/cdn/0.9.3//browserfs.min.js
_BFS_CACHE_KEY = "36fb7f27c808bd4880632a1d15a4be0c"
_BFS_CDN_URL   = "https://cdn.jsdelivr.net/npm/browserfs@1.4.3/dist/browserfs.min.js"

# Resolve the project root regardless of where this file lives inside the repo.
# os.path.dirname(__file__) is the SBS/ package directory; one level up is root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR   = os.path.join(_PROJECT_ROOT, "build", "web-cache")
_DATA_FILE   = os.path.join(_CACHE_DIR, f"{_BFS_CACHE_KEY}.data")
_HEAD_FILE   = os.path.join(_CACHE_DIR, f"{_BFS_CACHE_KEY}.head")

# Pygbag's testserver reads a .head file alongside every .data file.
# It parses "Key: Value" lines and sends them as HTTP response headers,
# stopping at the first line that does not contain ": ".
_HEAD_CONTENT = "Content-Type: application/javascript\n"


def _free_port(port: int):
    """
    Terminate any process currently bound to *port* so pygbag can claim it.

    Uses lsof (available on macOS and Linux) to find PIDs listening on the
    TCP port, then sends SIGTERM to each one.  Errors are printed but never
    raised — if the kill fails the subsequent pygbag bind will surface its
    own clear OSError rather than crashing here with a confusing traceback.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True
        )
        pids = [p for p in result.stdout.strip().split() if p]
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        if pids:
            print(f"[run_web] Freed port {port} "
                  f"(stopped PID(s): {', '.join(pids)}).")
    except Exception as exc:
        print(f"[run_web] Could not auto-free port {port}: {exc}")


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
    _free_port(8000)
    _ensure_browserfs()

    # Launch Pygbag as a child process so we can stream its output and act
    # after its build step.  os.execv cannot be used here because it replaces
    # this process entirely — we would have no opportunity to overwrite the
    # favicon that pygbag copies from its own defaults into build/web/.
    # --disable-sound-format-error: suppress the RuntimeError pygbag raises
    # when it finds .wav files inside .venv (pygame library examples) during
    # its directory scan — those files are never packaged into the final build.
    print("[run_web] Starting Pygbag — open http://localhost:8000\n")
    _proc = subprocess.Popen(
        [sys.executable, "-m", "pygbag",
         "--disable-sound-format-error", _PROJECT_ROOT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # Favicon guard: pygbag re-copies its default Python favicon on every build
    # cycle, overwriting whatever is in build/web/favicon.png.  A single copy
    # is not enough — we run a daemon thread that restores icon.png every second
    # so our favicon always survives, regardless of how many times pygbag rebuilds.
    _fav_src = os.path.join(_PROJECT_ROOT, "assets", "favicon.png")
    _fav_dst = os.path.join(_PROJECT_ROOT, "build", "web", "favicon.png")

    def _favicon_guard():
        while True:
            time.sleep(1)
            try:
                if os.path.isfile(_fav_src):
                    os.makedirs(os.path.dirname(_fav_dst), exist_ok=True)
                    shutil.copy2(_fav_src, _fav_dst)
            except Exception:
                pass

    threading.Thread(target=_favicon_guard, daemon=True).start()

    for _line in iter(_proc.stdout.readline, ""):
        sys.stdout.write(_line)
        sys.stdout.flush()

    _proc.wait()
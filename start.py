#!/usr/bin/env python3
"""
BiRAGAS Unified Application — One-Click Launcher
==================================================
Ayass Bioscience LLC

Double-click this file or run: python3 start.py

This will:
  1. Start the Python backend server (all 17 API endpoints)
  2. Open BiRAGAS_Unified_App.html in your browser
  3. All Demo buttons will work automatically
"""

import os
import sys
import time
import signal
import webbrowser
import subprocess
import threading

PORT = 8000
DIR = os.path.dirname(os.path.abspath(__file__))

def check_dependencies():
    """Check and install required packages."""
    required = ['fastapi', 'uvicorn', 'pydantic']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Done.")

def kill_existing():
    """Kill any existing server on the port."""
    try:
        import signal as sig
        result = subprocess.run(['lsof', '-ti', f':{PORT}'], capture_output=True, text=True)
        for pid in result.stdout.strip().split('\n'):
            if pid:
                os.kill(int(pid), sig.SIGTERM)
        time.sleep(1)
    except Exception:
        pass

def open_browser():
    """Open the app in the default browser after server starts."""
    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/status', timeout=2)
            print(f"\n  ✓ Server ready — opening browser...")
            webbrowser.open(f'http://127.0.0.1:{PORT}/')
            return
        except Exception:
            pass
    print("\n  Server did not start in time. Open http://127.0.0.1:8000/ manually.")

def main():
    print()
    print("=" * 60)
    print("  BiRAGAS Unified Application v1.0")
    print("  Ayass Bioscience LLC")
    print("=" * 60)
    print()
    print("  Starting backend server...")
    print(f"  Working directory: {DIR}")
    print()

    # Setup
    os.chdir(DIR)
    sys.path.insert(0, DIR)

    # Load .env
    env_path = os.path.join(DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
        print("  ✓ .env loaded")

    # Check deps
    check_dependencies()

    # Kill existing
    kill_existing()

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start server
    print(f"  Starting server on http://127.0.0.1:{PORT}/")
    print(f"  API docs: http://127.0.0.1:{PORT}/docs")
    print()
    print("  Press Ctrl+C to stop")
    print("-" * 60)

    try:
        from server import create_app
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Try: pip3 install fastapi uvicorn")
        input("\n  Press Enter to exit...")

if __name__ == "__main__":
    main()

#!/bin/sh
# Strata for an Apple silicon Mac (EXPERIMENTAL: the Metal engine, docs/APPLE_METAL.md).
# The first run compiles the engine, downloads the model and starts it; later runs just start it.
# Needs the Xcode command line tools (xcode-select --install), CMake and Python 3.10+.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || {
    echo "Python 3.10 or newer is needed (python.org, or: brew install python)."; exit 1; }
  python3 -m venv .venv || { rm -rf .venv; echo "could not create .venv"; exit 1; }
fi
.venv/bin/python -c 'import jinja2, regex, numpy, psutil' 2>/dev/null ||
  .venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt || {
    echo "the Python packages could not be installed (pip)"; exit 1; }
exec .venv/bin/python metal/setup_metal.py "$@"

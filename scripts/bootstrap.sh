#!/usr/bin/env bash
set -eu

# UNVERIFIED in M0: a minimal, POSIX-oriented convenience script.
python3 -m venv .venv
. .venv/bin/activate
python scripts/check_env.py
echo "Environment created. Follow docs/reproduction/environment.md before installing upstream dependencies."

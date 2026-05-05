#!/usr/bin/env bash
# Verilator strict lint over rtl/. Skips empty placeholder files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERILATOR_BIN="${VERILATOR_BIN:-verilator}"

if ! command -v "$VERILATOR_BIN" >/dev/null 2>&1; then
    echo "verilator not found on PATH"
    echo "install: apt-get install verilator (Linux) or pacman -S mingw-w64-x86_64-verilator (MSYS2)"
    echo "or override VERILATOR_BIN / VERILATOR_ROOT envvars"
    exit 1
fi

sv_files=()
for f in rtl/*.sv; do
    [[ -s "$f" ]] && sv_files+=("$f")
done

if (( ${#sv_files[@]} == 0 )); then
    echo "no non-empty rtl/*.sv to lint"
    exit 0
fi

# On Ubuntu, verilator package installs:
#   /usr/bin/verilator      (wrapper script)
#   /usr/bin/verilator_bin  (actual binary)
#   /usr/share/verilator/   (includes, examples)
# The wrapper script expects VERILATOR_ROOT=/usr/share/verilator
# but the binary path is hardcoded correctly.
# On MSYS2, verilator_bin.exe is in the same dir as the wrapper.
if [ -z "${VERILATOR_ROOT:-}" ]; then
    # Set VERILATOR_ROOT only if share dir exists and wrapper needs it
    if [ -d "/usr/share/verilator" ]; then
        export VERILATOR_ROOT="/usr/share/verilator"
    fi
fi

# Try running verilator directly; if it fails due to VERILATOR_ROOT,
# try with verilator_bin directly (Ubuntu package layout)
if ! "$VERILATOR_BIN" --lint-only -Wall --top-module tinyNPU_top "${sv_files[@]}" 2>/dev/null; then
    # Fallback: run verilator_bin directly if available (Ubuntu package)
    if command -v verilator_bin >/dev/null 2>&1; then
        echo "Falling back to verilator_bin directly..."
        verilator_bin --lint-only -Wall --top-module tinyNPU_top "${sv_files[@]}"
    else
        # Re-run to show the actual error
        "$VERILATOR_BIN" --lint-only -Wall --top-module tinyNPU_top "${sv_files[@]}"
    fi
fi

echo ">>> lint passed"

#!/usr/bin/env bash
# Verilator strict lint over rtl/. Skips empty placeholder files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# The verilator wrapper script can misbehave on both Ubuntu (wrong binary path)
# and MSYS2 (missing Perl Pod::Usage). Use verilator_bin directly everywhere.
UNAME_S="$(uname -s)"
if [[ "$UNAME_S" == Linux ]]; then
    VERILATOR_BIN="${VERILATOR_BIN:-verilator_bin}"
else
    VERILATOR_BIN="${VERILATOR_BIN:-verilator_bin.exe}"
fi

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

# VERILATOR_ROOT: needed so verilator can find its share/ includes.
# Derive from binary location if not set: <prefix>/bin/verilator -> <prefix>/share/verilator
if [ -z "${VERILATOR_ROOT:-}" ]; then
    bin_path="$(command -v "$VERILATOR_BIN")"
    bin_dir="$(dirname "$bin_path")"
    prefix="${bin_dir%/bin}"
    if [ -d "$prefix/share/verilator" ]; then
        export VERILATOR_ROOT="$prefix/share/verilator"
    elif [ -d "/usr/share/verilator" ]; then
        export VERILATOR_ROOT="/usr/share/verilator"
    fi
fi

"$VERILATOR_BIN" --lint-only -Wall --top-module tinyNPU_top "${sv_files[@]}"
echo ">>> lint passed"
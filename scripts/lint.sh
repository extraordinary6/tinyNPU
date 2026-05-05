#!/usr/bin/env bash
# Verilator strict lint over rtl/. Skips empty placeholder files.
# Uses verilator_bin.exe directly (the perl wrapper needs Pod::Usage which
# the MSYS2 system perl is missing). VERILATOR_ROOT must point to the
# installed share/ directory so verilated_std.sv can be found.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERILATOR_BIN="${VERILATOR_BIN:-verilator}"
export VERILATOR_ROOT="${VERILATOR_ROOT:-/usr/share/verilator}"

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

"$VERILATOR_BIN" --lint-only -Wall --top-module tinyNPU_top "${sv_files[@]}"
echo ">>> lint passed"

#!/usr/bin/env bash
# Run every cocotb testbench under tb/test_*. Intended for MSYS2 bash on Windows.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail=0
for tb_dir in tb/test_*/; do
    tb_name="${tb_dir%/}"
    [[ -f "$tb_name/Makefile" ]] || { echo "-- skip $tb_name (no Makefile)"; continue; }
    echo "=== $tb_name ==="
    if (cd "$tb_name" && make -s clean >/dev/null 2>&1; make); then
        echo "[ OK ] $tb_name"
    else
        echo "[FAIL] $tb_name"
        fail=$((fail + 1))
    fi
done

if (( fail > 0 )); then
    echo ">>> $fail testbench(es) failed"
    exit 1
fi
echo ">>> all testbenches passed"

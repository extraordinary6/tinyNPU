#!/usr/bin/env bash
# Run `make coverage` (Verilator + line/toggle coverage) for every testbench
# under tb/test_*, collect the per-tb coverage.info files into coverage_out/,
# and merge them into coverage_out/merged.info. A testbench that fails to
# build or run under Verilator is logged and skipped — partial coverage is
# better than none, and Verilator is stricter than Icarus so some failures
# are expected during initial bring-up.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/coverage_out"
rm -rf "$OUT"
mkdir -p "$OUT"

ok=0
skip=0
fail_list=()

for tb_dir in tb/test_*/; do
    tb_name="$(basename "${tb_dir%/}")"
    [[ -f "$tb_dir/Makefile" ]] || { echo "-- skip $tb_name (no Makefile)"; continue; }
    echo "=== $tb_name ==="

    (cd "$tb_dir" && make -s clean >/dev/null 2>&1; make coverage)
    rc=$?

    if [[ $rc -eq 0 && -f "$tb_dir/coverage.info" ]]; then
        cp "$tb_dir/coverage.info" "$OUT/${tb_name}.info"
        echo "[ OK ] $tb_name"
        ok=$((ok + 1))
    else
        echo "[FAIL] $tb_name (exit $rc, no coverage.info)"
        fail_list+=("$tb_name")
        skip=$((skip + 1))
    fi
done

echo
echo ">>> coverage produced for $ok testbench(es), skipped $skip"
if (( skip > 0 )); then
    echo ">>> failed: ${fail_list[*]}"
fi

if (( ok == 0 )); then
    echo ">>> no coverage.info files to merge"
    exit 1
fi

# lcov -a chains: -a a.info -a b.info -a c.info -o merged.info
add_args=()
for f in "$OUT"/*.info; do
    add_args+=(-a "$f")
done

if command -v lcov >/dev/null 2>&1; then
    lcov "${add_args[@]}" -o "$OUT/merged.info"
    echo ">>> wrote $OUT/merged.info"
else
    echo ">>> lcov not installed; per-tb info files left in $OUT"
fi

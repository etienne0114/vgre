#!/usr/bin/env bash
# VGRE footprint report (Track 20). Prints the on-disk size of the runtime
# libraries, both as built and stripped, so the lean (VGRE_MINIMAL) profile's
# footprint can be measured and tracked. Args: paths to the .so files.
set -u

echo "── VGRE footprint report ──"
total=0
for lib in "$@"; do
    [ -f "$lib" ] || continue
    bytes=$(stat -c %s "$lib" 2>/dev/null || echo 0)
    tmp=$(mktemp)
    cp "$lib" "$tmp" && strip --strip-unneeded "$tmp" 2>/dev/null || true
    sbytes=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
    rm -f "$tmp"
    awk -v n="$(basename "$lib")" -v b="$bytes" -v s="$sbytes" \
        'BEGIN{printf "  %-26s %8.1f MB   (stripped %7.1f MB)\n", n, b/1048576, s/1048576}'
    total=$((total + bytes))
done
awk -v t="$total" 'BEGIN{printf "  %-26s %8.1f MB\n", "TOTAL (as built)", t/1048576}'

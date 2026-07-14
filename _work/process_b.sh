#!/usr/bin/env bash
# Process dataset B one outer zip at a time, with cleanup between each to keep
# peak disk low. Deletes each outer drive-download zip ONLY after its images are
# successfully downscaled into navier_b/. Nested markers-*.zip parts are moved
# aside to _work/markers_parts/ for a separate final step.
set -u
ROOT="c:/Users/andre/Documents/ligmax-software"
cd "$ROOT" || exit 1
SEVENZIP="/c/Program Files/7-Zip/7z.exe"
DST="$ROOT/navier_b"
TMP="$ROOT/_work/b_tmp"
MARKERS="$ROOT/_work/markers_parts"
mkdir -p "$DST" "$MARKERS"

for outer in drive-download-*.zip; do
  echo "==================================================================="
  echo ">>> OUTER: $outer   ($(date))"
  df -h /c | tail -1
  rm -rf "$TMP"; mkdir -p "$TMP/extracted"

  # 1) extract outer -> inner zip
  unzip -o -j "$outer" -d "$TMP" >/dev/null 2>&1 || { echo "!! failed to unzip outer $outer"; exit 1; }
  inner=$(ls "$TMP"/*.zip 2>/dev/null | head -1)
  if [ -z "$inner" ]; then echo "!! no inner zip found in $outer"; exit 1; fi
  echo "    inner: $(basename "$inner")"

  # 2) extract inner -> image tree (+ possibly markers-*.zip)
  unzip -o "$inner" -d "$TMP/extracted" >/dev/null 2>&1 || { echo "!! failed to unzip inner"; exit 1; }

  # 3) move any nested markers-*.zip aside (don't lose them)
  found_markers=$(find "$TMP/extracted" -type f -iname '*.zip')
  if [ -n "$found_markers" ]; then
    while IFS= read -r mz; do
      echo "    setting aside markers part: $(basename "$mz")"
      mv -n "$mz" "$MARKERS"/ 2>/dev/null || true
    done <<< "$found_markers"
  fi

  # 4) downscale the extracted image tree into navier_b
  python _work/downscale_b.py "$TMP/extracted" "$DST"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!! downscale reported failures (rc=$rc) for $outer -- NOT deleting the zip"
    rm -rf "$TMP"
    exit 1
  fi

  # 5) success -> clean temp and delete the outer zip
  rm -rf "$TMP"
  rm -f "$outer" && echo "    deleted outer zip: $outer"
  echo "    navier_b size so far: $(du -sh "$DST" 2>/dev/null | cut -f1)"
done

rm -rf "$TMP"
echo "==================================================================="
echo ">>> ALL OUTER ZIPS PROCESSED"
echo "navier_b images: $(find "$DST" -type f -iname '*.jpg' | wc -l)"
echo "markers parts collected: $(ls -1 "$MARKERS" 2>/dev/null | wc -l)"
df -h /c | tail -1

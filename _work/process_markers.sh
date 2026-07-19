#!/usr/bin/env bash
# Process the 9 markers-*.zip parts one at a time into navier_b/.
# Each part is an independent zip. Extract -> downscale -> (only on verified
# success with >0 images) delete the extracted temp AND the markers zip.
set -u
ROOT="c:/Users/andre/Documents/ligmax-software"
cd "$ROOT" || exit 1
DST="$ROOT/navier_b"
TMP="$ROOT/_work/m_tmp"
MARKERS="$ROOT/_work/markers_parts"
mkdir -p "$DST"

for mz in "$MARKERS"/markers-*.zip; do
  echo "==================================================================="
  echo ">>> MARKERS: $(basename "$mz")   ($(date))"
  df -h /c | tail -1
  rm -rf "$TMP"; mkdir -p "$TMP"

  unzip -o "$mz" -d "$TMP" >/dev/null 2>&1 || { echo "!! unzip failed for $(basename "$mz")"; rm -rf "$TMP"; exit 1; }

  python _work/downscale_b.py "$TMP" "$DST"
  rc=$?
  rm -rf "$TMP"
  if [ "$rc" -ne 0 ]; then
    echo "!! downscale failed/empty (rc=$rc) for $(basename "$mz") -- KEEPING the zip"
    exit 1
  fi

  rm -f "$mz" && echo "    deleted markers zip: $(basename "$mz")"
  echo "    navier_b images so far: $(find "$DST" -type f -iname '*.jpg' | wc -l)  size: $(du -sh "$DST" 2>/dev/null | cut -f1)"
done

echo "==================================================================="
echo ">>> ALL MARKERS PROCESSED"
echo "navier_b total images: $(find "$DST" -type f -iname '*.jpg' | wc -l)"
echo "navier_b folders:"
find "$DST" -mindepth 1 -maxdepth 1 -type d | sort
df -h /c | tail -1

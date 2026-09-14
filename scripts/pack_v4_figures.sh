#!/usr/bin/env bash
# Pack every published figure/table out of a transfer_v4_final tree into one tarball.
#
# Picks up *.png and *.tex only, keeps the directory structure (best/, btl/,
# overlay_*/, venn_*/ ...) and skips the disposable _merged_* symlink trees, which
# hold nothing but links back into the read-only run dirs.
#
# Prints the tarball path as its LAST line of stdout, so a caller can capture it:
#   TGZ=$(scripts/pack_v4_figures.sh | tail -1)
#
# Usage:
#   scripts/pack_v4_figures.sh [SRC_DIR] [OUT_TGZ]
# Defaults:
#   SRC_DIR  /proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4_final
#   OUT_TGZ  /tmp/v4_figures_<YYYYmmdd-HHMMSS>.tgz
set -euo pipefail

SRC="${1:-/proj/MR_dataset/mtk53728/UQ/sep_scratch/transfer_v4_final}"
OUT="${2:-/tmp/v4_figures_$(date +%Y%m%d-%H%M%S).tgz}"

[ -d "$SRC" ] || { echo "no such dir: $SRC" >&2; exit 1; }

cd "$SRC"
# -print0 / --null: the OneDrive side has spaces in its paths, so never split on them.
find . \( -name '*.png' -o -name '*.tex' \) -not -path '*/_merged_*' -print0 \
    > /tmp/.v4_pack_list.$$
n=$(tr -cd '\0' < /tmp/.v4_pack_list.$$ | wc -c)
if [ "$n" -eq 0 ]; then
    echo "nothing to pack under $SRC" >&2
    rm -f /tmp/.v4_pack_list.$$
    exit 1
fi

tar --null -czf "$OUT" -T /tmp/.v4_pack_list.$$
rm -f /tmp/.v4_pack_list.$$

echo "packed $n files from $SRC  ($(du -h "$OUT" | cut -f1))" >&2
echo "$OUT"

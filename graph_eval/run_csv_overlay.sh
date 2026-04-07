#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_csv_overlay.sh \
    --plots "file.csv|column|label|color|style|width;file2.csv|column|label|color|style|width" \
    [--bins 160] [--xmin -4] [--xmax 4] [--normalize true] \
    [--ymin -1] [--ymax -1] [--draw-v0 true] [--sym-vline -1] \
    [--xtitle "Energy Resolution (%)"] [--ytitle "Normalized Events"] \
    [--title ""] [--legend-header ""] \
    [--output-root combined_output.root] [--tdir csv_overlay] \
    [--canvas-name overlay_canvas] [--png overlay.png]

Notes:
- Up to 8 plot specs are supported.
- color can be ROOT color token (kRed, kBlue, ...) or integer code.
- Use --normalize false to keep raw event counts.
USAGE
}

PLOTS=""
BINS=160
XMIN=-4
XMAX=4
NORMALIZE=true
YMIN=-1
YMAX=-1
DRAW_V0=true
SYM_VLINE=-1
XTITLE="Energy Resolution (%)"
YTITLE="Normalized Events"
TITLE=""
LEGEND_HEADER=""
OUTPUT_ROOT="combined_output.root"
TDIR="csv_overlay"
CANVAS_NAME="overlay_canvas"
PNG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plots) PLOTS="$2"; shift 2;;
    --bins) BINS="$2"; shift 2;;
    --xmin) XMIN="$2"; shift 2;;
    --xmax) XMAX="$2"; shift 2;;
    --normalize) NORMALIZE="$2"; shift 2;;
    --ymin) YMIN="$2"; shift 2;;
    --ymax) YMAX="$2"; shift 2;;
    --draw-v0) DRAW_V0="$2"; shift 2;;
    --sym-vline) SYM_VLINE="$2"; shift 2;;
    --xtitle) XTITLE="$2"; shift 2;;
    --ytitle) YTITLE="$2"; shift 2;;
    --title) TITLE="$2"; shift 2;;
    --legend-header) LEGEND_HEADER="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --tdir) TDIR="$2"; shift 2;;
    --canvas-name) CANVAS_NAME="$2"; shift 2;;
    --png) PNG="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "$PLOTS" ]]; then
  echo "ERROR: --plots is required."
  usage
  exit 1
fi

root -l -b -q "graph_eval/plot_csv_overlays.C(\"${PLOTS}\",${BINS},${XMIN},${XMAX},${NORMALIZE},${YMIN},${YMAX},${DRAW_V0},${SYM_VLINE},\"${XTITLE}\",\"${YTITLE}\",\"${TITLE}\",\"${LEGEND_HEADER}\",\"${OUTPUT_ROOT}\",\"${TDIR}\",\"${CANVAS_NAME}\",\"${PNG}\")"

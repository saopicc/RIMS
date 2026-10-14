#!/usr/bin/env bash
# CPU profile of `rims run` with NVIDIA Nsight Systems.
#
# Usage (from the directory you would normally run rims in):
#   /path/to/RIMS/profiling/run_nsys.sh <rims run arguments...>
#
# Prerequisites: `nsys` on PATH, and `pip install nvtx` (or `pip install -e .[profiling]`)
# in the environment providing `rims`, otherwise the RIMS NVTX ranges are absent.
#
# Output: rims_<branch>_<commit>_<date>.nsys-rep in $NSYS_OUTDIR (default: cwd).
# Open it in nsys-ui, or summarise it on the command line, e.g.
#   nsys stats -r nvtx_sum,osrt_sum <file>.nsys-rep
# Running this on the *_profiling branch of each code version with identical arguments
# gives reports whose NVTX ranges have the same names, so nvtx_sum tables compare directly.
set -euo pipefail

if ! command -v nsys >/dev/null; then
    echo "nsys not found on PATH" >&2
    exit 1
fi
if ! python -c "import nvtx" 2>/dev/null; then
    echo "WARNING: python package nvtx not installed, RIMS NVTX ranges will be missing" >&2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
OUT="${NSYS_OUTDIR:-$PWD}/rims_${BRANCH}_${COMMIT}_$(date +%Y%m%d-%H%M%S)"

# --trace-fork-before-exec: DDFacet's AsyncProcessPool forks its workers without exec,
#   these are only followed with this flag
# --sample/--cpuctxsw process-tree: CPU IP sampling and thread scheduling across all workers
# osrt: OS runtime calls (file I/O, waits, sleeps), to separate I/O from compute
nsys profile \
    --trace=nvtx,osrt \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --trace-fork-before-exec=true \
    --stats=false \
    --output="$OUT" \
    rims run "$@"

echo "Profile written to ${OUT}.nsys-rep"

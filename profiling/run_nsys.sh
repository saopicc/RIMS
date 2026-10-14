#!/usr/bin/env bash
# CPU profile of `rims run` with NVIDIA Nsight Systems.
#
# Usage (from the directory you would normally run rims in):
#   /path/to/RIMS/profiling/run_nsys.sh <rims run arguments...>
#
# Prerequisites: `nsys` on PATH (or NSYS=/path/to/nsys), and `pip install nvtx` (or `pip install -e .[profiling]`)
# in the environment providing `rims`, otherwise the RIMS NVTX ranges are absent.
#
# NSYS_TRACE (default: nvtx) is passed to `nsys profile --trace`. NSYS_TRACE=nvtx,osrt adds OS runtime
# calls (file I/O, waits), but with the jax kernels that is ~9 GB of raw trace per worker in $TMPDIR
# (or /tmp), and the report then takes hours to generate. Keep osrt for short or numpy-only runs.
#
# Output: rims_<branch>_<commit>_<date>.nsys-rep in $NSYS_OUTDIR (default: cwd).
# Open it in nsys-ui, or summarise it on the command line, e.g.
#   nsys stats -r nvtx_sum <file>.nsys-rep
# Running this on the *_profiling branch of each code version with identical arguments
# gives reports whose NVTX ranges have the same names, so nvtx_sum tables compare directly.
set -euo pipefail

NSYS=${NSYS:-nsys}  # set NSYS=/path/to/nsys if it is not on PATH
NSYS_TRACE=${NSYS_TRACE:-nvtx}
if ! command -v "$NSYS" >/dev/null; then
    echo "nsys not found (set NSYS=/path/to/nsys)" >&2
    exit 1
fi
if ! python -c "import nvtx" 2>/dev/null; then
    echo "WARNING: python package nvtx not installed, RIMS NVTX ranges will be missing" >&2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
# Run the checked-out code, not a (possibly stale, non-editable) installed copy of DynSpecMS
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
OUT="${NSYS_OUTDIR:-$PWD}/rims_${BRANCH}_${COMMIT}_$(date +%Y%m%d-%H%M%S)"

# CPU sampling needs perf_event_paranoid <= 2, otherwise only the $NSYS_TRACE traces are collected
SAMPLE=process-tree
if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 2 ]; then
    echo "WARNING: kernel.perf_event_paranoid > 2, CPU sampling and context switch tracing disabled" >&2
    SAMPLE=none
fi

# --trace-fork-before-exec: DDFacet's AsyncProcessPool forks its workers without exec,
#   these are only followed with this flag
# --sample/--cpuctxsw process-tree: CPU IP sampling and thread scheduling across all workers
"$NSYS" profile \
    --trace="$NSYS_TRACE" \
    --sample=$SAMPLE \
    --cpuctxsw=$SAMPLE \
    --trace-fork-before-exec=true \
    --stats=false \
    --output="$OUT" \
    rims run "$@"

echo "Profile written to ${OUT}.nsys-rep"

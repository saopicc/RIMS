"""
NVTX annotations for CPU profiling with NVIDIA Nsight Systems (nsys).

Ranges show up on the nsys timeline under the "RIMS" domain, one row per
process/thread (main process, I/O worker, compute workers). Everything here is
a no-op, returning the undecorated function, unless:
  - the `nvtx` package is installed (pip install nvtx), and
  - the environment variable RIMS_NVTX is not set to 0.
When nvtx is installed but no profiler is attached, the ranges cost ~nothing.

Usage:
    from DynSpecMS.Profiling import annotate, nvtx_range

    @annotate(color="blue")          # range named after Class.method
    def LoadMS(self, iJob): ...

    with nvtx_range("await LoadMS", color="grey"):
        ...

See profiling/run_nsys.sh for how to collect a profile.
"""
import os
import functools
import contextlib

DOMAIN = "RIMS"

# Colour per category, so related ranges group visually on the timeline
COLORS = {"io": "blue",        # reading the MS / catalogues
          "compute": "green",  # gridding / stacking
          "jones": "purple",   # beam and kMS Jones matrices
          "output": "orange",  # writing fits / zarr / catalogues / plots
          "setup": "yellow",   # initialisation, catalogue building
          "wait": "grey"}      # main process waiting on workers

try:
    import nvtx as _nvtx
    ENABLED = os.environ.get("RIMS_NVTX", "1") != "0"
except ImportError:
    _nvtx = None
    ENABLED = False


def annotate(name=None, color=None, category=None):
    """ Decorator wrapping the function in an NVTX range.
    name defaults to the function's __qualname__ (e.g. ClassDynSpecMS.LoadMS);
    category is a key of COLORS and is used when color is not given.
    """
    def decorator(func):
        if not ENABLED:
            return func
        attr = _nvtx.annotate(message=name or func.__qualname__,
                              color=color or COLORS.get(category),
                              domain=DOMAIN)

        @functools.wraps(func)  # keeps __name__, needed by AsyncProcessPool.runJob
        def wrapper(*args, **kwargs):
            with attr:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def nvtx_range(name, color=None, category=None):
    """ Context manager for an NVTX range around a block of code """
    if not ENABLED:
        return contextlib.nullcontext()
    return _nvtx.annotate(message=name, color=color or COLORS.get(category), domain=DOMAIN)

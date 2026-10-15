"""
Measurement Set access for RIMS.

All RIMS MS reads go through xarray-ms, which presents an MSv2 table as MSv4
(time, baseline_id, frequency, polarization) datasets read with arcae
(the DDFacet beam machinery in setJones still opens the MS itself).
Visibilities are loaded into shared memory as rows ordered by
(time, baseline) and preprocessed block by block in a thread pool
(arcae and numpy release the GIL).
"""
import functools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from DDFacet.Array import shared_dict
from DDFacet.Other import logger
log = logger.getLogger("MSReader")
from .Profiling import annotate

# xarray-ms decodes the MSv2 TIME (MJD seconds) to unix seconds; RIMS works
# in MJD seconds. Adding the offset back recovers the TIME column exactly
MJD_UNIX_OFFSET = 3506716800.0

# MSv2 column names that MSv4 renames, all other columns keep their MSv2 name.
# MSv4 WEIGHT is WEIGHT_SPECTRUM, or WEIGHT broadcast over channels if the MS
# has no WEIGHT_SPECTRUM
MSV2_TO_MSV4 = {"DATA": "VISIBILITY",
                "WEIGHT_SPECTRUM": "WEIGHT",
                "WEIGHT": "WEIGHT"}

# MSv4 polarization labels -> MSv2 CORR_TYPE codes
CORR_TYPE = {"RR": 5, "RL": 6, "LR": 7, "LL": 8,
             "XX": 9, "XY": 10, "YX": 11, "YY": 12}

# time steps per read block
BLOCK_TIMES = 4


def variable_name(ColName):
    return MSV2_TO_MSV4.get(ColName, ColName)


def open_ms(MSName, auto_corrs=False):
    """
    Partitions (DataTree nodes) of MSName sorted by time. Opening builds the
    MS structure, so it is cached: each MS is opened once per process and
    forked workers inherit the opened MS
    """
    return _open_ms(MSName, bool(auto_corrs))


@functools.lru_cache(maxsize=None)
@annotate(name="MSReader.open_ms", category="io")
def _open_ms(MSName, auto_corrs):
    dt = xr.open_datatree(MSName, engine="xarray-ms:msv2", auto_corrs=auto_corrs)
    return tuple(sorted(dt.children.values(), key=lambda node: node.ds.time.values[0]))


def ms_times(MSName, auto_corrs=False):
    """Unique times (MJD seconds) of MSName"""
    times = np.concatenate([node.ds.time.values for node in open_ms(MSName, auto_corrs)])
    return np.unique(times) + MJD_UNIX_OFFSET


def ms_phase_dir(MSName, auto_corrs=False):
    """Phase centre (ra, dec) in radians of the single field of MSName"""
    dirs = np.concatenate([node["field_and_source_base_xds"].ds.FIELD_PHASE_CENTER_DIRECTION.values
                           for node in open_ms(MSName, auto_corrs)])
    dirs = np.unique(dirs, axis=0)
    if dirs.shape[0] != 1:
        raise ValueError("%s has %i phase centres, a single field is needed" % (MSName, dirs.shape[0]))
    ra, dec = dirs[0]
    return ra, dec


def ms_metadata(MSName, auto_corrs=False):
    """Metadata of MSName needed by RIMS"""
    nodes = open_ms(MSName, auto_corrs)
    ds = nodes[0].ds
    for node in nodes[1:]:
        if not (np.array_equal(node.ds.frequency.values, ds.frequency.values) and
                np.array_equal(node.ds.polarization.values, ds.polarization.values)):
            raise ValueError("%s: partitions with different channels or correlations are not supported" % MSName)

    intervals = np.unique([node.ds.time.attrs["integration_time"]["data"] for node in nodes])
    if intervals.size != 1 or not np.isfinite(intervals[0]):
        raise ValueError("%s: irregular integration times %s" % (MSName, intervals))

    ChanWidth = ds.frequency.attrs["channel_width"]["data"]
    if not np.isfinite(ChanWidth):
        ChanWidth = ds.CHANNEL_WIDTH.values
    ChanWidth = np.abs(np.broadcast_to(ChanWidth, ds.frequency.shape))

    obs = ds.attrs["observation_info"]
    return {"ChanFreq": ds.frequency.values,
            "ChanWidth": ChanWidth,
            "times": ms_times(MSName, auto_corrs),
            "dtBin": intervals[0],
            "CorrType": [CORR_TYPE[p] for p in ds.polarization.values],
            "ra0dec0": ms_phase_dir(MSName, auto_corrs),
            "TELESCOPE_NAME": nodes[0]["antenna_xds"].ds.attrs["overall_telescope_name"],
            "OBSERVER": obs["observer"][0],
            "PROJECT": obs["project_UID"],
            "Variables": set(ds.data_vars)}


def antenna_indices(node):
    """ANTENNA table indices of the two antennas of each baseline of a partition"""
    index = {name: i for i, name in enumerate(node["antenna_xds"].ds.antenna_name.values)}
    A0 = np.array([index[name] for name in node.ds.baseline_antenna1_name.values], np.int32)
    A1 = np.array([index[name] for name in node.ds.baseline_antenna2_name.values], np.int32)
    return A0, A1


@annotate(name="MSReader.load_chunk", category="io")
def load_chunk(DicoName, MSName, tmin, tmax, ColName, ModelName=None, ColWeights=None,
               PolIndices=(0, 1, 2, 3), RevertChans=False, UVRange=(1., 1000.), NThreads=8):
    """
    Reads the rows of MSName with tmin <= TIME < tmax (MJD seconds) into a
    new shared dict DicoName, ordered by (time, baseline):
      data    (nrow, nch, npol) complex64: ColName - ModelName, zero where flagged
      flag    (nrow, nch, npol) bool: FLAG, set for rows outside UVRange (km)
      weights (nrow, nch) float32: ColWeights averaged over correlations, or ones
      times, A0, A1 (nrow,); u, v, w (nrow, 1, 1); uniq_times
    Correlations are selected by PolIndices and channels reversed if
    RevertChans. Grid cells missing from the MS are flagged.
    Returns the shared dict, or None if there is no data in the interval
    """
    uv0, uv1 = np.array(UVRange) * 1000
    chSlice = slice(None, None, -1) if RevertChans else slice(None)
    PolIndices = list(PolIndices)
    npol = len(PolIndices)

    # (variables, time slice, first row) of each partition overlapping the interval
    Parts = []
    times, A0, A1, uvw = [], [], [], []
    nrow = 0
    # autocorrelations have |uvw| = 0, so they are only read if the uv-cut keeps them
    for node in open_ms(MSName, auto_corrs=uv0 <= 0):
        ds = node.ds
        ptimes = ds.time.values + MJD_UNIX_OFFSET
        it = np.where((ptimes >= tmin) & (ptimes < tmax))[0]
        if it.size == 0: continue
        tslice = slice(it[0], it[-1] + 1)
        a0, a1 = antenna_indices(node)
        nch = ds.sizes["frequency"]

        PolSel = slice(None) if PolIndices == list(range(ds.sizes["polarization"])) else PolIndices
        Vars = {"vis": ds[variable_name(ColName)].isel(polarization=PolSel),
                "model": ds[variable_name(ModelName)].isel(polarization=PolSel) if ModelName else None,
                "weights": ds[variable_name(ColWeights)] if ColWeights else None,
                "flag": ds.FLAG.isel(polarization=PolSel)}
        Parts.append((Vars, tslice, nrow))

        times.append(np.repeat(ptimes[tslice], a0.size))
        A0.append(np.tile(a0, it.size))
        A1.append(np.tile(a1, it.size))
        uvw.append(ds.UVW.isel(time=tslice).values.reshape((-1, 3)))
        nrow += it.size * a0.size

    if nrow == 0:
        return None

    times, A0, A1, uvw = (np.concatenate(x) for x in (times, A0, A1, uvw))
    # missing grid cells are flagged; zero their nan uvw so that it can't reach the phases
    uvw[np.isnan(uvw)] = 0.
    u, v, w = uvw.T
    d = np.sqrt(u**2 + v**2 + w**2)
    UVCut = (d < uv0) | (d > uv1)

    DicoDATA = shared_dict.create(DicoName)
    data = DicoDATA.addSharedArray("data", (nrow, nch, npol), np.complex64)
    flag = DicoDATA.addSharedArray("flag", (nrow, nch, npol), np.bool_)
    weights = DicoDATA.addSharedArray("weights", (nrow, nch), np.float32)

    # blocks of BLOCK_TIMES time steps: (variables, time slice, row range)
    Tasks = []
    for Vars, tslice, row0 in Parts:
        nbl = Vars["vis"].sizes["baseline_id"]
        for t0 in range(tslice.start, tslice.stop, BLOCK_TIMES):
            t1 = min(t0 + BLOCK_TIMES, tslice.stop)
            r0 = row0 + (t0 - tslice.start) * nbl
            Tasks.append((Vars, slice(t0, t1), r0, r0 + (t1 - t0) * nbl))

    @annotate(name="MSReader.read_block", category="io")
    def fill(task):
        Vars, tslice, r0, r1 = task
        n = r1 - r0
        d, f, wt = data[r0:r1], flag[r0:r1], weights[r0:r1]

        vis = Vars["vis"].isel(time=tslice).values.reshape((n, nch, npol))
        if Vars["model"] is None:
            np.copyto(d, vis[:, chSlice])
        else:
            model = Vars["model"].isel(time=tslice).values.reshape((n, nch, npol))
            np.subtract(vis[:, chSlice], model[:, chSlice], out=d)
            del model
        del vis

        if Vars["weights"] is None:
            wt.fill(1)
        else:
            ws = Vars["weights"].isel(time=tslice).values
            ws = ws.reshape((n,) + ws.shape[2:])
            if ws.ndim == 3:
                np.mean(ws[:, chSlice], axis=-1, out=wt)
            else:
                np.copyto(wt, ws[:, chSlice])
            del ws

        fl = Vars["flag"].isel(time=tslice).values.reshape((n, nch, npol))
        np.not_equal(fl[:, chSlice], 0, out=f)
        f[UVCut[r0:r1]] = True
        np.copyto(d, 0, where=f)
        return n

    T0 = time.time()
    with ThreadPoolExecutor(max(1, NThreads)) as Pool:
        for _ in Pool.map(fill, Tasks): pass
    dT = time.time() - T0
    print("  Read %i rows (%.1f GB in shared memory) in %.1fs with %i threads" %
          (nrow, (data.nbytes + flag.nbytes + weights.nbytes) / 1e9, dT, NThreads), file=log)

    DicoDATA["times"] = times
    DicoDATA["A0"] = A0
    DicoDATA["A1"] = A1
    DicoDATA["u"] = u.reshape((-1, 1, 1))
    DicoDATA["v"] = v.reshape((-1, 1, 1))
    DicoDATA["w"] = w.reshape((-1, 1, 1))
    DicoDATA["uniq_times"] = np.unique(times)
    return DicoDATA

"""
Tests of DynSpecMS.MSReader (xarray-ms) against the casacore reading it
replaced, on small simulated Measurement Sets in a temporary directory.

Run with: python -m pytest tests
"""
import os
import shutil

import numpy as np
import pytest
from casacore.tables import table
from xarray_ms.testing.simulator import MSStructureSimulator

from DynSpecMS import MSReader

NANT, NCHAN, NTIME = 5, 16, 12
CORRS = ["XX", "XY", "YX", "YY"]

VARIABLE_SHAPE_COMPLEX = {"_c_order": True,
                          "dataManagerGroup": "StandardStMan",
                          "dataManagerType": "StandardStMan",
                          "keywords": {},
                          "maxlen": 0,
                          "ndim": 2,
                          "option": 0,
                          "valueType": "COMPLEX"}
EXTRA_COLUMNS = {"CORRECTED_DATA": {**VARIABLE_SHAPE_COMPLEX, "comment": "CORRECTED_DATA column"},
                 "MODEL_DATA": {**VARIABLE_SHAPE_COMPLEX, "comment": "MODEL_DATA column"}}

# (name, ColName, ModelName, ColWeights, PolIndices, RevertChans, UVRange)
CASES = [
    ("standard", "CORRECTED_DATA", "MODEL_DATA", "WEIGHT_SPECTRUM", (0, 1, 2, 3), False, (1., 1000.)),
    ("IQ pols", "CORRECTED_DATA", "MODEL_DATA", "WEIGHT_SPECTRUM", (0, 3), False, (1., 1000.)),
    ("reversed chans", "CORRECTED_DATA", "MODEL_DATA", "WEIGHT_SPECTRUM", (0, 1, 2, 3), True, (1., 1000.)),
    ("DATA, no model or weights", "DATA", "", None, (0, 1, 2, 3), False, (1., 1000.)),
    ("autocorrelations kept", "CORRECTED_DATA", "MODEL_DATA", "WEIGHT_SPECTRUM", (0, 1, 2, 3), False, (0., 1000.)),
    ("uv cut", "CORRECTED_DATA", None, "WEIGHT_SPECTRUM", (0, 3), True, (0.5, 2.)),
]


def realistic_data(missing_baseline):
    """
    Replaces the simulator ramps with random visibilities, per-correlation
    weights, flags (with nan visibilities under some flags) and uvw.
    If missing_baseline, baseline (0, 1) is absent from the second chunk
    """
    def transform(chunk_desc, data_dict):
        rng = np.random.default_rng(chunk_desc.chunk_id)
        a1, a2 = data_dict["ANTENNA1"][1], data_dict["ANTENNA2"][1]
        dims = data_dict["DATA"][0]
        shape = (a1.size, NCHAN, len(CORRS))

        def vis():
            return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

        flag = rng.random(shape) < 0.3
        corrected = vis()
        corrected[flag & (rng.random(shape) < 0.5)] = np.nan
        uvw = rng.uniform(-3000., 3000., (a1.size, 3))
        uvw[a1 == a2] = 0.
        data_dict.update(DATA=(dims, vis()),
                         CORRECTED_DATA=(dims, corrected),
                         MODEL_DATA=(dims, vis()),
                         WEIGHT_SPECTRUM=(dims, rng.uniform(0.5, 2., shape).astype(np.float32)),
                         FLAG=(dims, flag),
                         UVW=(data_dict["UVW"][0], uvw))

        if missing_baseline and chunk_desc.chunk_id == 1:
            keep = ~((a1 == 0) & (a2 == 1))
            data_dict = {k: (d, a[keep]) for k, (d, a) in data_dict.items()}
        return data_dict
    return transform


@pytest.fixture(scope="module", params=[False, True], ids=["regular", "missing baseline"])
def simulated_ms(request, tmp_path_factory):
    name = str(tmp_path_factory.mktemp("ms") / "simulated.ms")
    MSStructureSimulator(ntime=NTIME, time_chunks=4, nantenna=NANT,
                         data_description=[(NCHAN, CORRS)],
                         auto_corrs=True, simulate_data=True,
                         table_desc=EXTRA_COLUMNS,
                         transform_data=realistic_data(request.param)).simulate_ms(name)
    return name


@pytest.fixture(scope="session", autouse=True)
def cleanup_shared_memory():
    """DDFacet shared dicts live in /dev/shm/ddf.<pid> of this process"""
    yield
    shutil.rmtree("/dev/shm/ddf.%i" % os.getpid(), ignore_errors=True)


def casacore_load(MSName, tmin, tmax, ColName, ModelName, ColWeights, PolIndices, RevertChans, UVRange):
    """ClassDynSpecMS.LoadMS before xarray-ms (4b84040), reading with casacore"""
    t = table(MSName, ack=False, readonly=True)
    times = t.getcol("TIME")
    ind = np.where((times >= tmin) & (times < tmax))[0]
    NROW, ROW0 = ind.size, ind[0]
    nch = table("%s::SPECTRAL_WINDOW" % MSName, ack=False).getcol("CHAN_FREQ").ravel().size
    npol = table("%s::POLARIZATION" % MSName, ack=False).getcol("NUM_CORR").flat[0]
    PolIndices = list(PolIndices)

    data = np.zeros((NROW, nch, npol), np.complex64)
    t.getcolnp(ColName, data, ROW0, NROW)
    if RevertChans: data = data[:, ::-1, :]
    data = data[:, :, PolIndices]
    if ModelName:
        model = np.zeros((NROW, nch, npol), np.complex64)
        t.getcolnp(ModelName, model, ROW0, NROW)
        if RevertChans: model = model[:, ::-1, :]
        data -= model[:, :, PolIndices]
    if ColWeights:
        weights = np.zeros((NROW, nch, npol), np.float32)
        t.getcolnp(ColWeights, weights, ROW0, NROW)
        weights = np.mean(weights, axis=-1)
        if RevertChans: weights = weights[:, ::-1]
    else:
        weights = np.ones((NROW, nch), np.float32)
    flag = np.zeros((NROW, nch, npol), bool)
    t.getcolnp("FLAG", flag, ROW0, NROW)
    if RevertChans: flag = flag[:, ::-1]
    flag = flag[:, :, PolIndices]
    times = t.getcol("TIME", ROW0, NROW)
    A0, A1 = t.getcol("ANTENNA1", ROW0, NROW), t.getcol("ANTENNA2", ROW0, NROW)
    u, v, w = t.getcol("UVW", ROW0, NROW).T
    t.close()
    d = np.sqrt(u**2 + v**2 + w**2)
    uv0, uv1 = np.array(UVRange) * 1000
    flag[np.where((d < uv0) | (d > uv1))[0], :, :] = 1
    data[flag] = 0
    return dict(data=data, weights=weights, flag=flag, times=times, A0=A0, A1=A1,
                u=u.reshape((-1, 1, 1)), v=v.reshape((-1, 1, 1)), w=w.reshape((-1, 1, 1)))


def ms_times(MSName):
    return np.unique(table(MSName, ack=False, readonly=True).getcol("TIME"))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("window", [(0, NTIME), (2, 9)], ids=["all times", "time chunk"])
def test_load_chunk_matches_casacore(simulated_ms, case, window):
    _, ColName, ModelName, ColWeights, PolIndices, RevertChans, UVRange = case
    times = ms_times(simulated_ms)
    tmin = times[window[0]]
    tmax = times[window[1]] if window[1] < times.size else times[-1] + 1.
    ref = casacore_load(simulated_ms, tmin, tmax, ColName, ModelName, ColWeights,
                        PolIndices, RevertChans, UVRange)

    DicoDATA = MSReader.load_chunk("test_load_chunk", simulated_ms, tmin, tmax, ColName,
                                   ModelName=ModelName, ColWeights=ColWeights,
                                   PolIndices=PolIndices, RevertChans=RevertChans,
                                   UVRange=UVRange, NThreads=4)
    try:
        new = {k: np.array(DicoDATA[k]) for k in list(ref) + ["uniq_times"]}
    finally:
        DicoDATA.delete()

    for k in ref:
        assert new[k].dtype == ref[k].dtype, k
        assert new[k].shape[1:] == ref[k].shape[1:], k
    np.testing.assert_array_equal(new["uniq_times"], np.unique(ref["times"]))

    # casacore rows -> rows read with xarray-ms, by (time, antenna1, antenna2)
    rows = {key: r for r, key in enumerate(zip(new["times"], new["A0"], new["A1"]))}
    idx = np.array([rows.get(key, -1) for key in zip(ref["times"], ref["A0"], ref["A1"])])
    both = idx >= 0
    for k in ref:
        np.testing.assert_array_equal(new[k][idx[both]], ref[k][both], err_msg=k)

    # rows that only casacore read are autocorrelations removed by the uv-cut
    assert np.all(ref["A0"][~both] == ref["A1"][~both])
    assert ref["flag"][~both].all() and not ref["data"][~both].any()

    # (time, baseline) cells that are not in the MS are flagged, with zero data and uvw
    only_new = np.setdiff1d(np.arange(new["times"].size), idx[both])
    assert new["flag"][only_new].all() and not new["data"][only_new].any()
    for k in "uvw":
        assert not new[k][only_new].any()


def test_no_data_in_interval(simulated_ms):
    times = ms_times(simulated_ms)
    assert MSReader.load_chunk("test_no_data", simulated_ms, times[-1] + 1., times[-1] + 2.,
                               "DATA") is None


def test_metadata_matches_casacore(simulated_ms):
    Infos = MSReader.ms_metadata(simulated_ms)
    t = table(simulated_ms, ack=False, readonly=True)
    spw = table("%s::SPECTRAL_WINDOW" % simulated_ms, ack=False)
    obs = table("%s::OBSERVATION" % simulated_ms, ack=False)

    np.testing.assert_array_equal(Infos["ChanFreq"], spw.getcol("CHAN_FREQ").ravel())
    np.testing.assert_array_equal(Infos["ChanWidth"], np.abs(spw.getcol("CHAN_WIDTH").ravel()))
    np.testing.assert_array_equal(Infos["times"], np.unique(t.getcol("TIME")))
    assert Infos["dtBin"] == np.unique(t.getcol("INTERVAL")).item()
    assert Infos["CorrType"] == table("%s::POLARIZATION" % simulated_ms, ack=False).getcol("CORR_TYPE").ravel().tolist()
    np.testing.assert_array_equal(Infos["ra0dec0"], table("%s::FIELD" % simulated_ms, ack=False).getcol("PHASE_DIR").ravel())
    assert Infos["TELESCOPE_NAME"] == obs.getcol("TELESCOPE_NAME")[0]
    assert Infos["OBSERVER"] == obs.getcol("OBSERVER")[0]
    assert Infos["PROJECT"] == obs.getcol("PROJECT")[0]
    assert {"VISIBILITY", "CORRECTED_DATA", "MODEL_DATA", "WEIGHT", "FLAG"} <= Infos["Variables"]


def test_open_ms_is_cached(simulated_ms):
    # positional, keyword and numpy bool arguments share one open MS
    assert MSReader.open_ms(simulated_ms, False) is MSReader.open_ms(simulated_ms, auto_corrs=np.bool_(False))
    assert MSReader.open_ms(simulated_ms, True) is not MSReader.open_ms(simulated_ms, False)

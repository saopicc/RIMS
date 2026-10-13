"""
- jones_diag_for_time(G, tm, freq_domains, chan_freqs, time_value, dir_idx)
    -> J (nChan, nDir, nAnt) complex64
    Diagonal Jones term J[0,0] of the Jones solution time nearest to
    `time_value`, expanded from Jones frequency domains to visibility channels
    and selected for the Jones direction `dir_idx[k]` attached to each target k.

Done in numpy on the worker: only the (nFreqJones, nAnt, nDirJones) slice of
the shared G array for the current time is touched, so the full Jones cube is
never copied per time step.

Assumptions about G, tm and freq_domains (match what's produced by
ClassDynSpecMS.setJones in DicoJones):
- G shape == (nTimeJones, nFreqJones, nAnt, nDirJones, 2, 2)
- tm shape == (nTimeJones,)
- freq_domains shape == (nFreqJones, 2) of (nu0, nu1) in Hz
"""
from __future__ import division

import numpy as np


def jones_diag_for_time(G, tm, freq_domains, chan_freqs, time_value, dir_idx):
    """
    Parameters
    ----------
    G : (nTimeJones, nFreqJones, nAnt, nDirJones, 2, 2) complex
    tm : (nTimeJones,) Jones time centroids
    freq_domains : (nFreqJones, 2) Jones frequency domains [nu0, nu1) in Hz
    chan_freqs : (nChan,) visibility channel frequencies in Hz
    time_value : scalar, same units as tm
    dir_idx : (nDir,) int, Jones direction index for each target direction

    Returns
    -------
    J : (nChan, nDir, nAnt) complex64
        J[c, k, a] is the product of G[iT, iF, a, dir_idx[k], 0, 0] over all
        frequency domains iF that contain channel c (in practice exactly one).
        Channels outside every domain get 1, i.e. are left uncorrected.
    """
    chan_freqs = np.asarray(chan_freqs).ravel()
    freq_domains = np.asarray(freq_domains).reshape((-1, 2))
    dir_idx = np.asarray(dir_idx)

    iTJones = np.argmin(np.abs(tm - time_value))
    # (nFreqJones, nDir, nAnt) - only the slice for this time step is read
    G00 = np.ascontiguousarray(G[iTJones, :, :, :, 0, 0][:, :, dir_idx].transpose((0, 2, 1)))

    # (nFreqJones, nChan) channel membership of each frequency domain
    inDomain = (chan_freqs[None, :] >= freq_domains[:, 0:1]) & (chan_freqs[None, :] < freq_domains[:, 1:2])
    nDomPerChan = inDomain.sum(axis=0)

    if np.all(nDomPerChan <= 1):
        # usual case: domains don't overlap, a single gather does it
        iDom = np.argmax(inDomain, axis=0)
        J = G00[iDom].astype(np.complex64)
        J[nDomPerChan == 0] = 1.
    else:
        # overlapping domains are applied one after the other, as in the
        # original per-domain loop
        nDir, nAnt = G00.shape[1:]
        J = np.ones((chan_freqs.size, nDir, nAnt), np.complex64)
        for iFJones in np.where(inDomain.any(axis=1))[0]:
            J[inDomain[iFJones]] *= G00[iFJones]
    return J

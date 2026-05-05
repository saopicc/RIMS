"""
Numba implementation of time indexing and Jones diagnostic extract
"""
from __future__ import division

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _build_channel_domain_mapping_numba(chan_freqs, freq_domains):
    nChan = chan_freqs.shape[0]
    nFd = freq_domains.shape[0]
    ch_idx = np.empty(nChan, dtype=np.int32)
    ch_idx.fill(-1)
    if nFd == 0:
        return ch_idx

    for i in range(nChan):
        f = chan_freqs[i]
        for j in range(nFd):
            if f >= freq_domains[j,0] and f < freq_domains[j,1]:
                ch_idx[i] = j
                break
    return ch_idx

@njit(fastmath=True, cache=True)
def compute_jones_diag_for_time_numba(G, tm, freq_domains, chan_freqs, time_value, default_val=1.0+0.0j):
    """
    Compute per-antenna, per-channel diagonal Jones J_00 for nearest Jones time.
    """
    time_diffs = np.abs(tm - time_value)
    iTJones = np.argmin(time_diffs)

    G_time = G[iTJones]  # (nFreqJones, nAnt, nDirJones, 2, 2)
    
    nAnt = G_time.shape[1]
    nDirJones = G_time.shape[2]
    nChan = chan_freqs.shape[0]

    ch_domain_idx = _build_channel_domain_mapping_numba(chan_freqs, freq_domains)

    J_diag = np.empty((nAnt, nChan, nDirJones), dtype=np.complex128)

    for a in range(nAnt):
        for d in range(nDirJones):
            for c in range(nChan):
                fd_idx = ch_domain_idx[c]
                if fd_idx >= 0:
                    J_diag[a, c, d] = G_time[fd_idx, a, d, 0, 0]
                else:
                    J_diag[a, c, d] = default_val

    return J_diag, iTJones, ch_domain_idx

"""
Numba-enabled kernel to phase visibilities to directions and sum them.
"""
import numpy as np
from numba import njit, prange

C = 299792458.0

@njit(fastmath=True, cache=True)
def radec2lm_numba(ra, dec, ra0, dec0):
    l = np.cos(dec) * np.sin(ra - ra0)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return l, m

@njit(fastmath=True, cache=True)
def phase_and_sum_direction_numba(
    vis, flag, weights, u, v, w, A0s, A1s, chan_freqs,
    ra_all, dec_all, ra0, dec0,
    slicePol_idx, 
    J_diag=None, iDJones_all=None
):
    nrow = vis.shape[0]
    nch = vis.shape[1]
    npol = vis.shape[2]
    
    nDir = ra_all.shape[0]
    nOutPol = slicePol_idx.shape[0]

    ds = np.zeros((nDir, nch, nOutPol), dtype=np.complex64)
    ws = np.zeros((nDir, nch, nOutPol), dtype=np.float32)
    w2s = np.zeros((nDir, nch, nOutPol), dtype=np.float32)
    
    do_jones = (J_diag is not None) and (iDJones_all is not None)

    for iDir in prange(nDir):
        ra = ra_all[iDir]
        dec = dec_all[iDir]
        
        l, m = radec2lm_numba(ra, dec, ra0, dec0)
        n = np.sqrt(max(1.0 - l*l - m*m, 0.0))
        
        idx_jones_dir = 0
        if do_jones:
            idx_jones_dir = iDJones_all[iDir]
        
        for i_row in range(nrow):
            u_row = u[i_row,0,0]
            v_row = v[i_row,0,0]
            w_row = w[i_row,0,0]

            uvw_dot = u_row * l + v_row * m + w_row * (n - 1.0)
            
            a0 = A0s[i_row]
            a1 = A1s[i_row]

            for i_ch in range(nch):
                freq = chan_freqs[i_ch]
                kterm = -2.0j * np.pi * freq / C
                phase = np.exp(kterm * uvw_dot)
                
                # Jones correction
                if do_jones:
                    j0 = np.conj(J_diag[a0, i_ch, idx_jones_dir])
                    j1 = J_diag[a1, i_ch, idx_jones_dir]
                    jcorr = j0 * j1
                    wjcorr = (np.abs(j0) * np.abs(j1))**2
                else:
                    jcorr = 1.0 + 0.0j
                    wjcorr = 1.0

                w_base = weights[i_row, i_ch]
                
                for out_p in range(nOutPol):
                    p = slicePol_idx[out_p]
                    if not flag[i_row, i_ch, p]:
                        w_eff = w_base * wjcorr
                        val = vis[i_row, i_ch, p] * jcorr * phase
                        
                        ds[iDir, i_ch, out_p] += val
                        ws[iDir, i_ch, out_p] += w_eff
                        w2s[iDir, i_ch, out_p] += w_eff * w_eff

    return ds, ws, w2s

"""
- compute_jones_diag_for_time(DicoJones, chan_freqs, time_value, default_val=1+0j)
    -> (J_diag, iTJones, ch_domain_idx)
    Compute per-antenna, per-channel diagonal Jones (J_00) for the nearest
    Jones solution time to `time_value`. J_diag has shape
    (nAnt, nChan, nDirJones) and is returned as a JAX array (jnp.ndarray).

- extract_row_jones_jax(J_diag, A0s, A1s, dir_idx)
    -> (J0, J1, ch_indices)
    Given J_diag, produce per-row, per-channel J0/J1 arrays of shape
    (nRow, nChan, 1) usable directly inside a JAX phasing kernel. dir_idx
    can be a scalar (same Jones direction for all rows) or an array (one per row).

Assumptions about DicoJones (match what's produced by ClassDynSpecMS.setJones):
- DicoJones['G'] shape == (nTimeJones, nFreqJones, nAnt, nDirJones, 2, 2)
  (so indexing like G[iTJones, iFJones, Aindex, iDJones, 0, 0] is valid)
- DicoJones['tm'] shape == (nTimeJones,)
- DicoJones['FreqDomains'] is an iterable/list/array of (nu0, nu1) in Hz,
  length nFreqJones

Usage sketch:
    J_diag, iT = compute_jones_diag_for_time(DicoJones, chan_freqs, ThisTime)
    J0, J1, ch_mask = extract_row_jones_jax(J_diag, A0s, A1s, dir_idx)
    # Pass (J0, J1, ch_mask) into your phase_and_sum kernel (which can be jitted).
"""
from __future__ import division

import jax
import jax.numpy as jnp
from jax import jit, vmap
JAX = True

@jit
def _build_channel_domain_mapping(chan_freqs : jnp.ndarray, freq_domains : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized mapping of each channel to a Jones freq-domain index.

    - chan_freqs: (nChan,)
    - freq_domains: array-like (nFreqJones, 2) of (nu0, nu1)

    Returns:
    - ch_domain_idx: (nChan,) int32, index of domain for each channel or -1
    - ch_any_mask: (nChan,) bool indicates channels matched to any domain
    """
    chan_freqs = jnp.asarray(chan_freqs)
    fd = jnp.asarray(freq_domains)
    if fd.size == 0:
        nChan = chan_freqs.shape[0]
        return -jnp.ones((nChan,), dtype=jnp.int32), jnp.zeros((nChan,), dtype=bool)

    nu0 = fd[:, 0].reshape((-1, 1))
    nu1 = fd[:, 1].reshape((-1, 1))
    
    mask_all = (chan_freqs.reshape((1, -1)) >= nu0) & (chan_freqs.reshape((1, -1)) < nu1)
    
    ch_any = jnp.any(mask_all, axis=0)
    
    ch_idx = jnp.argmax(mask_all, axis=0).astype(jnp.int32)
    
    ch_idx = jnp.where(ch_any, ch_idx, -1)
    return ch_idx, ch_any


@jit
def compute_jones_diag_for_time(DicoJones, chan_freqs, time_value, default_val=1+0j):
    """
    Compute per-antenna, per-channel diagonal Jones J_00 for nearest Jones time.

    Parameters
    ----------
    DicoJones : dict-like (supports indexing with keys)
        'G' : (nTimeJones, nFreqJones, nAnt, nDirJones, 2, 2)
        'tm': (nTimeJones,)
        'FreqDomains' : iterable/lists of (nu0, nu1) length nFreqJones
    chan_freqs : array-like (nChan,) frequencies in Hz
    time_value : scalar (same units as DicoJones['tm'])
    default_val : complex scalar used for channels not covered by any domain

    Returns
    -------
    J_diag : jnp.ndarray shape (nAnt, nChan, nDirJones) complex
        The diagonal element J[0,0] for each antenna, channel, and Jones direction.
    iTJones : int
        Index of the selected Jones time slice (nearest to time_value).
    ch_domain_idx : jnp.ndarray shape (nChan,) int32
        Index of the freq-domain each channel was assigned to, or -1.
    """

    G = jnp.asarray(DicoJones["G"])            # (nT, nFdom, nAnt, nDirJones, 2, 2)
    tm = jnp.asarray(DicoJones["tm"])          # (nT,)
    freq_domains = jnp.asarray(DicoJones["FreqDomains"])  # (nFdom, 2)
    chan_freqs = jnp.asarray(chan_freqs)

    # 1) choose nearest Jones time index
    # Using argmin on abs difference
    time_diffs = jnp.abs(tm - time_value)
    iTJones = jnp.argmin(time_diffs).astype(jnp.int32)

    # 2) slice the time => G_time shape (nFreqJones, nAnt, nDirJones, 2, 2)
    G_time = G[iTJones]  # (nFreqJones, nAnt, nDirJones, 2, 2)

    # 3) extract diagonal (0,0) per freq-domain -> (nFreqJones, nAnt, nDirJones)
    J00_domains = G_time[:, :, :, 0, 0]

    # 4) determine channel -> freq-domain mapping
    # ch_domain_idx shape (nChan,), -1 if not in any domain
    ch_domain_idx, ch_any_mask = _build_channel_domain_mapping(chan_freqs, freq_domains)

    # 5) safe gather: for channels with -1 put index 0 temporarily then overwrite with default
    safe_idx = jnp.where(ch_domain_idx >= 0, ch_domain_idx, 0)  # (nChan,)

    # gather J00 for each channel using safe_idx -> shape (nChan, nAnt, nDirJones)
    # then transpose to (nAnt, nChan, nDirJones)
    # Use jnp.take along axis 0 to gather domains
    J_chan_ant_dir = jnp.take(J00_domains, safe_idx, axis=0)  # (nChan, nAnt, nDirJones)
    J_chan_ant_dir = jnp.transpose(J_chan_ant_dir, (1, 0, 2))  # (nAnt, nChan, nDirJones)

    # replace channels with default_val where ch_any_mask is False
    if jnp.any(~ch_any_mask):
        # build default block
        nAnt = J_chan_ant_dir.shape[0]
        nChan = J_chan_ant_dir.shape[1]
        nDirJones = J_chan_ant_dir.shape[2]
        default_block = jnp.full((nAnt, nChan, nDirJones), default_val, dtype=J_chan_ant_dir.dtype)
        mask_broadcast = (~ch_any_mask).reshape((1, nChan, 1))
        J_diag = jnp.where(mask_broadcast, default_block, J_chan_ant_dir)
    else:
        J_diag = J_chan_ant_dir

    return J_diag, iTJones, ch_domain_idx

@jit
def _select_dir_slice(J_diag : jnp.ndarray, dir_idx : int) -> jnp.ndarray:
    """
    Select a single direction slice from J_diag along axis 2.
    J_diag: (nAnt, nChan, nDirJones)
    dir_idx: scalar int
    returns J_dir: (nAnt, nChan)
    """
    return J_diag[:, :, dir_idx]


def extract_row_jones_jax(J_diag : jnp.ndarray, A0s : jnp.ndarray, A1s : jnp.ndarray, dir_idx : int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build per-row per-channel J0 and J1 arrays suitable for immediate use.

    Parameters
    ----------
    J_diag : jnp.ndarray (nAnt, nChan, nDirJones)
    A0s, A1s : array-like (nRow,) antenna indices per row
    dir_idx : scalar int or array-like (nRow,) Jones-direction index per row

    Returns
    -------
    J0 : jnp.ndarray (nRow, nChan, 1) complex
    J1 : jnp.ndarray (nRow, nChan, 1) complex
    ch_indices : jnp.ndarray (nChan,) int  (0..nChan-1)
    """
    J_diag = jnp.asarray(J_diag)
    A0s = jnp.asarray(A0s, dtype=jnp.int32)
    A1s = jnp.asarray(A1s, dtype=jnp.int32)

    nRow = A0s.shape[0]
    nChan = J_diag.shape[1]

    # Case 1: scalar dir_idx for all rows -> simple indexing
    if jnp.ndim(dir_idx) == 0:
        dir_idx = int(dir_idx)  # convert DeviceArray scalar to int for take
        J_dir = _select_dir_slice(J_diag, dir_idx)  # (nAnt, nChan)
        # Use advanced indexing on antenna axis to select per-row antennas
        # J_dir[A0s, :] -> (nRow, nChan)
        J0 = J_dir[A0s, :]
        J1 = J_dir[A1s, :]

    else:
        # dir_idx is per-row array; we need J_diag[A_ant, :, d_idx] for each row
        dir_idx = jnp.asarray(dir_idx, dtype=jnp.int32)  # (nRow,)

        # We'll vmapped over rows to gather row-specific (nChan,) arrays
        def _row_fetch(a_idx, d_idx):
            # returns shape (nChan,)
            return J_diag[a_idx, :, d_idx]

        vmap_row_fetch = vmap(_row_fetch, in_axes=(0, 0), out_axes=0)
        J0 = vmap_row_fetch(A0s, dir_idx)   # (nRow, nChan)
        J1 = vmap_row_fetch(A1s, dir_idx)   # (nRow, nChan)

    # Add trailing singleton axis to match kernel broadcasting expectations
    J0 = J0[..., jnp.newaxis]   # (nRow, nChan, 1)
    J1 = J1[..., jnp.newaxis]   # (nRow, nChan, 1)
    ch_indices = jnp.arange(nChan, dtype=jnp.int32)
    return J0, J1, ch_indices


# ----------------------
# Example integration helper (non-jitted, small convenience)
# ----------------------
def example_usage(DicoJones, chan_freqs, ThisTime, A0s, A1s, iDJones_for_dirs):
    """
    Convenience example showing the flow. Not jitted.

    - Compute J_diag once per time:
        J_diag, iTJones, ch_domain_idx = compute_jones_diag_for_time(...)
    - For each iDir you can get J0,J1 quickly:
        J0, J1, ch_indices = extract_row_jones_jax(J_diag, A0s, A1s, dir_idx)

    Note: We return JAX arrays; if you need NumPy arrays call .block_until_ready() and np.array(...)
    """
    J_diag, iTJones, ch_domain_idx = compute_jones_diag_for_time(DicoJones, chan_freqs, ThisTime)
    # Example: for a particular direction index (iDJones_for_dirs[idir]):
    some_dir = int(iDJones_for_dirs[0])
    J0, J1, ch_idx = extract_row_jones_jax(J_diag, A0s, A1s, some_dir)
    return J0, J1, ch_idx
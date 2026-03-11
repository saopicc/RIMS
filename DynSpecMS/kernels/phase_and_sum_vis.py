"""
JAX-enabled kernel to phase visibilities to directions and sum them.

Behavior:
- Phases visibilities to a target (ra,dec) given a phase centre (ra0,dec0).
- Optionally applies Jones (gain) corrections (precomputed or via a callable).
- Sums visibilities across rows -> returns per-channel, per-polarisation sums
  and weight sums (and weight-squared sums) compatible with existing code.

Usage sketch:
  ds, ws, w2s = phase_and_sum_direction(vis, flag, weights,
                                        u, v, w, A0s, A1s,
                                        chan_freqs, ra, dec, ra0, dec0,
                                        slicePol=slice(None),
                                        Jones=None)
"""
from __future__ import division
import jax.numpy as jnp
from jax import jit
JAX_AVAILABLE = True
from typing import Optional

import numpy as np

# speed of light (m/s)
C = 299792458.0


def radec2lm(ra : jnp.ndarray, dec: jnp.ndarray, ra0: jnp.ndarray, dec0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert RA/DEC to direction cosines l,m for a given phase centre.

    Parameters
    ----------
    ra, dec, ra0, dec0 : scalar or array-like (radians)
        Input coordinates.

    Returns
    -------
    l, m : same array type as inputs (jnp or np)
    """
    # using the same formula as ClassDynSpecMS.radec2lm
    l = jnp.cos(dec) * jnp.sin(ra - ra0)
    m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(ra - ra0)
    return l, m


@jit
def _compute_phase(chfreq : jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray, n: jnp.ndarray):
    """Compute phasor exp(-2pi i nu/c * (u*l + v*m + w*(n-1)))
    
    Parameters
    ----------
    chfreq : array_like, shape (nch,), float
        Channel centre frequencies in Hz.
    u, v, w : array_like, shape (nrow, 1, 1)
        Baseline coordinates in metres.
    l, m, n : scalar or array-like
        Direction cosines.
    Returns
    -------
    phase : array_like, shape (nrow, nch, 1), complex
        Phasor values.
    """
    chf = chfreq.reshape((1, -1, 1))
    kterm = -2.0 * jnp.pi * 1j * chf / C
    uvw_dot = u * l + v * m + w * (n - 1.0)
    return jnp.exp(kterm * uvw_dot)


def phase_and_sum_direction(vis : jnp.ndarray, flag : jnp.ndarray, weights : jnp.ndarray, u : jnp.ndarray, v : jnp.ndarray, w : jnp.ndarray, A0s : jnp.ndarray, A1s : jnp.ndarray,
                            chan_freqs : jnp.ndarray, ra : float, dec : float, ra0 : float, dec0 : float,
                            slicePol=slice(None), Jones=Optional[dict])-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase visibilities to a direction, apply optional Jones corrections, and sum.

    Parameters
    ----------
    vis : array_like, shape (nrow, nch, npol), complex
        Visibilities for selected time rows.
    flag : array_like, shape (nrow, nch, npol), bool
        Flag array (True => flagged).
    weights : array_like, shape (nrow, nch) or (nrow, nch, 1)
        Weights (scalar per visibility, broadcasted to pols).
    u, v, w : array_like, shape (nrow,) or (nrow,1,1)
        Baseline coordinates in metres.
    A0s, A1s : array_like, shape (nrow,)
        Antenna indices per visibility row (used for Jones lookups if provided).
    chan_freqs : array_like, shape (nch,), float
        Channel centre frequencies in Hz.
    ra, dec : float
        Target direction (radians).
    ra0, dec0 : float
        Phase centre (radians).
    slicePol : slice or sequence
        Polarisation mapping (like self.slicePol).
    Jones : tuple of arrays with J0 and J1
        tuple (J0, J1, ch_mask) : J0,J1 are arrays broadcastable to
        (nrow, n_ch_mask, 1) and ch_mask selects channels they apply to.

    Returns
    -------
    ds : ndarray (nch, noutpol) complex (numpy)
    ws : ndarray (nch, noutpol) float  (numpy)
    w2s: ndarray (nch, noutpol) float  (numpy)
    """
    nrow, nch, npol = vis.shape

    # build per-pol weights array
    W = jnp.zeros((nrow, nch, npol), dtype=jnp.float32)
    # broadcast the (nrow,nch) weights to each pol
    for i_p in range(npol):
        W = W.at[:, :, i_p].set(weights)

    # zero weights where flagged
    W = jnp.where(flag, 0.0, W)

    dcorr = jnp.asarray(vis)

    # compute l,m,n
    l, m = radec2lm(ra, dec, ra0, dec0)
    n = jnp.sqrt(jnp.clip(1.0 - l * l - m * m, a_min=0.0))

    # compute phasing
    phase = _compute_phase(chan_freqs, u, v, w, l, m, n)

    if Jones is not None:
        J0, J1, ch_mask = Jones

        dcorr = dcorr.at[:, ch_mask, :].set(J0.conj() * dcorr[:, ch_mask, :] * J1)
        W = W.at[:, ch_mask, :].set(W[:, ch_mask, :] * (jnp.abs(J0) * jnp.abs(J1)) ** 2)

    # Apply phase and sum across rows
    dcorr = dcorr * phase
    ds = jnp.sum(dcorr, axis=0)    # (nch, npol)
    ws = jnp.sum(W, axis=0)
    w2s = jnp.sum(W ** 2, axis=0)

    # Apply requested polarisation mapping
    if isinstance(slicePol, slice):
        ds_out = ds[:, slicePol]
        ws_out = ws[:, slicePol]
        w2s_out = w2s[:, slicePol]
    else:
        # assume slicePol is sequence of indices
        idx = list(slicePol)
        ds_out = ds[:, idx]
        ws_out = ws[:, idx]
        w2s_out = w2s[:, idx]

    return np.array(ds_out), np.array(ws_out), np.array(w2s_out)
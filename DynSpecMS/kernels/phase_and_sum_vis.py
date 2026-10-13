"""
JAX kernel to phase the visibilities of one time step to every target
direction and sum them.

Behavior (identical to the per-direction loop of the non-jax implementation):
- Weights the visibilities (flagged weights set to zero).
- Optionally applies the diagonal Jones term: conj(J[A0]) * V * J[A1], with
  the weights scaled by |J[A0]|^2 |J[A1]|^2.
- Phases to each direction (l, m, n) and sums over rows, returning
  per-direction, per-channel, per-polarisation sums of the visibilities,
  the weights and the squared weights.

The kernel loops over channels (lax.map) and does all directions at once for
each channel, so the (ndir, nrow) phase and Jones terms of a channel stay in
cache and the sum over rows is a small (ndir, nrow) x (nrow, npol) matmul.

Usage sketch:
  ds, ws, w2s = phase_and_sum_directions(vis, flag, weights, u, v, w,
                                         chan_freqs, l, m, n,
                                         J=None, A0s=None, A1s=None)
"""
from __future__ import division
import warnings
import numpy as np

# Importing DDFacet's AsyncProcessPool switches on flush-to-zero, so numpy
# warns that the smallest subnormal is zero whenever finfo is first built for
# a dtype. Build jax's finfo here, before the workers are forked, so the
# workers don't each print this harmless warning on their first jax call.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
    import jax
    # float64 is needed for the phase range reduction (and for u, v, w, freqs)
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit
    jnp.finfo(np.float32), jnp.finfo(np.float64)

# speed of light (m/s), same value as astropy.constants.c
C = 299792458.0

# rows are padded to a multiple of this so that time steps with a different
# number of rows don't each trigger a recompilation
ROW_BLOCK = 128


@jit
def _phase_and_sum(vis, flag, weights, u, v, w, chan_freqs, l, m, n, J, A0s, A1s):
    nrow, nch, npol = vis.shape
    ndir = l.shape[0]

    # Per-pol weights, zero where flagged, channel-major for the channel loop
    W = jnp.where(flag, jnp.float32(0), weights[:, :, None])  # (nrow, nch, npol) float32
    VW = jnp.transpose(vis * W, (1, 0, 2))                    # (nch, nrow, npol) complex64
    W = jnp.transpose(W, (1, 0, 2))                           # (nch, nrow, npol) float32

    uvw_dot = u[None, :] * l[:, None] + v[None, :] * m[:, None] + w[None, :] * (n[:, None] - 1.)  # (ndir, nrow) float64

    def phase(fc):
        # Phase in turns, range-reduced in float64 so the float32 sin/cos is
        # accurate even for phases of thousands of radians
        turns = uvw_dot * (fc / C)
        turns = turns - jnp.round(turns)
        ang = (-2. * np.pi * turns).astype(jnp.float32)
        return jax.lax.complex(jnp.cos(ang), jnp.sin(ang))    # (ndir, nrow) complex64

    if J is None:
        def one_chan(x):
            fc, VWc = x
            return jnp.matmul(phase(fc), VWc)                 # (ndir, npol)
        ds = jax.lax.map(one_chan, (chan_freqs, VW))          # (nch, ndir, npol)
        ws = jnp.broadcast_to(jnp.sum(W, axis=1)[:, None, :], (nch, ndir, npol))
        w2s = jnp.broadcast_to(jnp.sum(W**2, axis=1)[:, None, :], (nch, ndir, npol))
    else:
        def one_chan(x):
            fc, Jc, VWc, Wc = x                               # Jc (ndir, nAnt)
            J0 = Jc[:, A0s]                                   # (ndir, nrow)
            J1 = Jc[:, A1s]
            A = (jnp.abs(J0) * jnp.abs(J1))**2                # (ndir, nrow) float32
            ds = jnp.matmul(phase(fc) * (J0.conj() * J1), VWc)
            ws = jnp.matmul(A, Wc)
            w2s = jnp.matmul(A**2, Wc**2)
            return ds, ws, w2s
        ds, ws, w2s = jax.lax.map(one_chan, (chan_freqs, J, VW, W))

    # (nch, ndir, npol) -> (ndir, nch, npol)
    return [jnp.transpose(x, (1, 0, 2)) for x in (ds, ws, w2s)]


def phase_and_sum_directions(vis, flag, weights, u, v, w, chan_freqs, l, m, n, J=None, A0s=None, A1s=None):
    """
    Phase visibilities to every direction, apply optional Jones corrections, and sum.

    Parameters
    ----------
    vis : array_like, shape (nrow, nch, npol), complex
        Visibilities for selected time rows (zero where flagged).
    flag : array_like, shape (nrow, nch, npol), bool
        Flag array (True => flagged).
    weights : array_like, shape (nrow, nch)
        Weights (scalar per visibility, broadcasted to pols).
    u, v, w : array_like, shape (nrow,)
        Baseline coordinates in metres.
    chan_freqs : array_like, shape (nch,), float
        Channel centre frequencies in Hz.
    l, m, n : array_like, shape (ndir,)
        Direction cosines of the directions to phase to.
    J : array_like, shape (nch, ndir, nAnt), complex, or None
        Diagonal Jones term per channel, direction and antenna
        (see t_idx_jones.jones_diag_for_time).
    A0s, A1s : array_like, shape (nrow,)
        Antenna indices per visibility row (only used with J).

    Returns
    -------
    ds : ndarray (ndir, nch, npol) complex64
    ws : ndarray (ndir, nch, npol) float32
    w2s: ndarray (ndir, nch, npol) float32
    """
    nrow, nch, npol = vis.shape
    nrow_pad = -(-nrow // ROW_BLOCK) * ROW_BLOCK

    def pad(a, val=0):
        a = np.asarray(a)
        if nrow_pad == nrow: return a
        return np.concatenate([a, np.full((nrow_pad - nrow,) + a.shape[1:], val, a.dtype)])

    # padded rows are flagged with zero weight, so they add nothing to the sums
    args = (pad(np.asarray(vis, np.complex64)),
            pad(np.asarray(flag, bool), True),
            pad(np.asarray(weights, np.float32)),
            pad(np.asarray(u, np.float64).ravel()),
            pad(np.asarray(v, np.float64).ravel()),
            pad(np.asarray(w, np.float64).ravel()),
            np.asarray(chan_freqs, np.float64).ravel(),
            np.asarray(l, np.float64), np.asarray(m, np.float64), np.asarray(n, np.float64))
    if J is None:
        Jargs = (None, None, None)
    else:
        Jargs = (np.asarray(J, np.complex64),
                 pad(np.asarray(A0s, np.int32).ravel()),
                 pad(np.asarray(A1s, np.int32).ravel()))
    ds, ws, w2s = _phase_and_sum(*args, *Jargs)
    return np.asarray(ds), np.asarray(ws), np.asarray(w2s)

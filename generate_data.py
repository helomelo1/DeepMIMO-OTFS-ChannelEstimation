"""
MIMO-OTFS Dataset Generation (CPU / NumPy)
Paper: "Deep Learning-based Channel Estimation for Massive MIMO-OTFS Communication Systems"
Payami & Blostein, WTS 2024

Run:
    pip install numpy h5py tqdm
    python generate_dataset.py
"""

import numpy as np
import h5py
from tqdm import tqdm
import time
import os

# ── System parameters ──────────────────────────────────────────────────────────
FC       = 2.15e9
V_MAX    = 360 / 3.6        # m/s
C        = 3e8

M_TAU    = 8                # delay pilot bins
N_NU     = 18               # Doppler pilot bins
NT       = 16               # transmit antennas at BS
NQ       = 6                # EVA multipath components
N_EL     = M_TAU * N_NU * NT  # total sparse vector length = 2304

TAU_R    = 1.0 / 15e3       # delay resolution (s)
NU_R     = 15e3 / 10        # Doppler resolution (Hz)

# EVA 3GPP channel model
EVA_DELAYS_SEC = np.array([0, 30, 150, 310, 370, 710]) * 1e-9
EVA_POWERS_LIN = 10 ** (np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1]) / 10)
EVA_POWERS_LIN = EVA_POWERS_LIN / EVA_POWERS_LIN.sum()

# Fixed delay bin indices (EVA delays are deterministic)
TAU_IDX = np.round(EVA_DELAYS_SEC / TAU_R).astype(int).clip(0, M_TAU - 1)

# ── Dataset config ─────────────────────────────────────────────────────────────
N_SAMPLES   = 100_000
BATCH_SIZE  = 1_000
SNR_DB      = 10.0
SAVE_PATH   = "mimo_otfs_dataset.h5"
TRAIN_SPLIT = 0.7


# ── Sensing matrix (built once, fixed for all samples) ─────────────────────────
def build_sensing_matrix(seed=42):
    rng  = np.random.default_rng(seed)
    rows = M_TAU * N_NU
    cols = N_EL
    Phi  = (rng.standard_normal((rows, cols)) +
            1j * rng.standard_normal((rows, cols))
           ).astype(np.complex64) / np.sqrt(2 * rows)
    PhiH = Phi.conj().T.copy()   # precompute Hermitian transpose
    return Phi, PhiH


# ── Batched channel generation (eq. 8 of the paper) ───────────────────────────
def generate_batch(batch_size, rng):
    """
    Vectorised: no Python loop over samples.
    Returns H_ADD of shape (batch_size, M_TAU, N_NU, NT).
    """
    # Random Doppler bin per path per sample
    nu_idx = rng.integers(0, N_NU, size=(batch_size, NQ))

    # Random angle of departure per path per sample
    phi = rng.uniform(-np.pi / 2, np.pi / 2, size=(batch_size, NQ))

    # ULA steering vector: a_t[n] = exp(j pi sin(phi) n)
    n   = np.arange(NT, dtype=np.float32)
    a_t = np.exp(1j * np.pi * np.sin(phi)[..., None] * n).astype(np.complex64)
    # a_t shape: (batch, NQ, NT)

    # Complex path gains scaled by EVA tap power
    real = rng.standard_normal((batch_size, NQ)).astype(np.float32)
    imag = rng.standard_normal((batch_size, NQ)).astype(np.float32)
    h_q  = (real + 1j * imag) * np.sqrt(EVA_POWERS_LIN / 2).astype(np.float32)
    # h_q shape: (batch, NQ)

    scale = np.sqrt(NT / NQ, dtype=np.float32)

    H_ADD = np.zeros((batch_size, M_TAU, N_NU, NT), dtype=np.complex64)

    for q in range(NQ):
        t_idx  = int(TAU_IDX[q])       # fixed delay bin for this path
        n_idx  = nu_idx[:, q]          # (batch,) Doppler bins
        gain   = h_q[:, q] * scale     # (batch,)
        contrib = gain[:, None] * a_t[:, q, :]   # (batch, NT)

        # Scatter contribution into H_ADD
        sample_idx = np.arange(batch_size)
        H_ADD[sample_idx, t_idx, n_idx, :] += contrib

    return H_ADD


# ── Observation vector y_DD = Phi h_ADD + noise (eq. 9) ───────────────────────
def build_observations(h_flat, Phi, snr_db, rng):
    """
    h_flat : (batch, N_EL) complex
    returns : (batch, ROWS) complex
    """
    y_clean   = (Phi @ h_flat.T).T                           # (batch, ROWS)
    snr_lin   = 10 ** (snr_db / 10)
    sig_power = np.mean(np.abs(y_clean) ** 2, axis=1, keepdims=True)
    noise_std = np.sqrt(sig_power / (2.0 * snr_lin))
    noise     = noise_std * (rng.standard_normal(y_clean.shape) +
                             1j * rng.standard_normal(y_clean.shape)
                            ).astype(np.complex64)
    return (y_clean + noise).astype(np.complex64)


# ── PositionNet input: |Phi^H y_DD| ───────────────────────────────────────────
def build_position_input(y_DD, PhiH):
    """
    Returns real-valued similarity vector of shape (batch, N_EL).
    """
    return np.abs((PhiH @ y_DD.T).T).astype(np.float32)


# ── Main generation loop ───────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  MIMO-OTFS Dataset Generator (NumPy CPU)")
    print("=" * 52)
    print(f"  Samples     : {N_SAMPLES:,}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  N_EL        : {N_EL}  ({M_TAU}×{N_NU}×{NT})")
    print(f"  SNR         : {SNR_DB} dB")
    print(f"  Output      : {SAVE_PATH}")
    print("=" * 52)

    rng = np.random.default_rng(42)

    print("\nBuilding sensing matrix Phi...")
    Phi, PhiH = build_sensing_matrix(seed=42)
    print(f"  Phi shape  : {Phi.shape}  ({Phi.nbytes/1e6:.1f} MB)")
    print(f"  PhiH shape : {PhiH.shape} ({PhiH.nbytes/1e6:.1f} MB)")

    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nGenerating {N_SAMPLES:,} samples in {n_batches} batches...\n")
    t0 = time.time()

    ROWS = M_TAU * N_NU   # = 144  (much smaller than N_EL=2304)

    with h5py.File(SAVE_PATH, "w") as f:
        # y_DD: (N_SAMPLES, 144)  complex64 — small, recompute X_position at train time
        ds_y  = f.create_dataset("y_DD",      shape=(N_SAMPLES, ROWS),
                                  dtype="complex64", chunks=(BATCH_SIZE, ROWS),
                                  compression="gzip", compression_opts=4)
        # h_ADD: (N_SAMPLES, N_EL) complex64 — stored as complex, not split
        ds_Yh = f.create_dataset("h_ADD",     shape=(N_SAMPLES, N_EL),
                                  dtype="complex64", chunks=(BATCH_SIZE, N_EL),
                                  compression="gzip", compression_opts=4)
        # binary support: (N_SAMPLES, N_EL) — compresses extremely well
        ds_Yb = f.create_dataset("Y_binary",  shape=(N_SAMPLES, N_EL),
                                  dtype="float32", chunks=(BATCH_SIZE, N_EL),
                                  compression="gzip", compression_opts=4)

        for i in tqdm(range(n_batches), desc="Generating"):
            start = i * BATCH_SIZE
            bs    = min(BATCH_SIZE, N_SAMPLES - start)

            H_ADD  = generate_batch(bs, rng)           # (bs, Mτ, Nν, Nt)
            h_flat = H_ADD.reshape(bs, -1)             # (bs, N_EL) complex64
            y_DD   = build_observations(h_flat, Phi, SNR_DB, rng)  # (bs, 144)
            binary = (np.abs(h_flat) > 1e-9).astype(np.float32)

            ds_y[start:start+bs]  = y_DD
            ds_Yh[start:start+bs] = h_flat
            ds_Yb[start:start+bs] = binary

    elapsed = time.time() - t0
    size_mb = os.path.getsize(SAVE_PATH) / 1e6

    print(f"\n✓ Done in {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  File size  : {size_mb:.1f} MB")

    # ── Verify and print split sizes ──────────────────────────────────────────
    with h5py.File(SAVE_PATH, "r") as f:
        X  = f["X_position"][:]
        Yb = f["Y_binary"][:]
        Yh = f["Y_channel"][:]

    split = int(TRAIN_SPLIT * N_SAMPLES)
    print(f"\n  Train : {split:,} samples")
    print(f"  Val   : {N_SAMPLES - split:,} samples")
    print(f"  X_position shape : {X.shape}")
    print(f"  Y_binary   shape : {Yb.shape}")
    print(f"  Y_channel  shape : {Yh.shape}")
    print(f"  Avg non-zeros/sample : {Yb.mean() * N_EL:.1f}  (expect ~{NQ})")
    print(f"\n✓ Dataset ready at: {os.path.abspath(SAVE_PATH)}")


if __name__ == "__main__":
    main()
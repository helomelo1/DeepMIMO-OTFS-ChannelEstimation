"""
Data generation for PositionNet (TensorFlow / Keras).

Paper: "Deep Learning-based Channel Estimation for
        Massive MIMO-OTFS Communication Systems"
        (Payami & Blostein, WTS 2024)

System model (Section II):
    y_DD = Phi @ h_ADD + z_DD
    PositionNet input = |Phi^H @ y_DD|  reshaped to (Mt, Nv, Nt, 1)
    PositionNet label = binary support mask (1 where |h| > 0)

Channel model: Extended Vehicular A (EVA) with ULA steering vectors.
Generates 10^5 samples, saves as .npz for tf.data loading.
"""

import numpy as np
from time import time

# ── System parameters (Section IV) ──────────────────────────────────────────
FC = 2.15e9                     # carrier frequency
V_MAX = 360 / 3.6              # max velocity m/s
SUBCARRIER_SPACING = 15e3      # 15 KHz

M_TAU = 150                    # delay pilot bins (Mr)
N_NU = 10                      # Doppler pilot bins (Nv)
NT = 16                        # transmit antennas (Nt)
NQ = 6                         # number of multipath scatterers (Nq)

N_EL = M_TAU * N_NU * NT      # 2304 total channel coefficients
ROWS = M_TAU * N_NU            # 144 pilot measurements

# ── EVA delay profile (6 taps) ──────────────────────────────────────────────
EVA_DELAYS_SEC = np.array([0, 30, 150, 310, 370, 710]) * 1e-9
EVA_POWERS_DB = np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1])
EVA_POWERS_LIN = 10 ** (EVA_POWERS_DB / 10)
EVA_POWERS_LIN /= EVA_POWERS_LIN.sum()

M_GRID = 600
TAU_R = 1.0 / (M_GRID * SUBCARRIER_SPACING)
TAU_IDX = np.round(EVA_DELAYS_SEC / TAU_R).astype(int).clip(0, M_TAU - 1)

# ── Dataset config ──────────────────────────────────────────────────────────
N_SAMPLES = 100_000
BATCH_SIZE = 1000               # large batches for vectorized speed
SNR_DB = 10.0
SAVE_PATH = "dataset_tf.npz"


def build_sensing_matrix(seed=42):
    """Complex Gaussian sensing matrix Phi (eq. 9)."""
    rng = np.random.default_rng(seed)
    Phi = (rng.standard_normal((ROWS, N_EL)) +
           1j * rng.standard_normal((ROWS, N_EL))
           ).astype(np.complex64) / np.sqrt(2 * ROWS)
    PhiH = Phi.conj().T.copy()
    return Phi, PhiH


def generate_channels(B, rng):
    """Generate B sparse channel realizations H_ADD using EVA + ULA model.

    Each path q has:
      - fixed delay index from EVA profile
      - random Doppler index in [0, N_NU)
      - random AoD in [-pi/2, pi/2]
      - complex gain ~ CN(0, EVA_POWERS[q])

    Returns H of shape (B, M_TAU, N_NU, NT).
    """
    nu_idx = rng.integers(0, N_NU, size=(B, NQ))
    phi_aod = rng.uniform(-np.pi / 2, np.pi / 2, size=(B, NQ))

    # ULA steering vectors: a_t(phi) = exp(j * pi * sin(phi) * n)
    n = np.arange(NT, dtype=np.float32)
    a_t = np.exp(1j * np.pi * np.sin(phi_aod)[..., None] * n).astype(np.complex64)

    # Complex path gains
    h_q = ((rng.standard_normal((B, NQ)) +
            1j * rng.standard_normal((B, NQ))) *
           np.sqrt(EVA_POWERS_LIN / 2)).astype(np.complex64)

    scale = np.sqrt(NT / NQ, dtype=np.float32)

    H = np.zeros((B, M_TAU, N_NU, NT), dtype=np.complex64)
    for q in range(NQ):
        gain = (h_q[:, q] * scale)[:, None] * a_t[:, q]  # (B, NT)
        H[np.arange(B), int(TAU_IDX[q]), nu_idx[:, q], :] += gain

    return H


def generate_observations(h_flat, Phi, snr_db, rng):
    """y_DD = Phi @ h + AWGN noise at given SNR."""
    y_clean = (Phi @ h_flat.T).T                              # (B, ROWS)

    snr_lin = 10 ** (snr_db / 10)
    sig_power = np.mean(np.abs(y_clean) ** 2, axis=1, keepdims=True)
    noise_std = np.sqrt(sig_power / (2 * snr_lin))

    noise = noise_std * (
        rng.standard_normal(y_clean.shape) +
        1j * rng.standard_normal(y_clean.shape)
    ).astype(np.complex64)

    return (y_clean + noise).astype(np.complex64)


def main():
    print(f"Generating {N_SAMPLES:,} samples (SNR={SNR_DB} dB)...\n")
    t0 = time()

    rng = np.random.default_rng(42)
    Phi, PhiH = build_sensing_matrix()

    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    # Pre-allocate output arrays — channels-last for TF: (B, Mt, Nv, Nt, 1)
    all_X = np.empty((N_SAMPLES, M_TAU, N_NU, NT, 1), dtype=np.float32)
    all_Y = np.empty((N_SAMPLES, M_TAU, N_NU, NT, 1), dtype=np.float32)

    for i in range(n_batches):
        start = i * BATCH_SIZE
        bs = min(BATCH_SIZE, N_SAMPLES - start)

        H = generate_channels(bs, rng)
        h_flat = H.reshape(bs, -1)

        y_DD = generate_observations(h_flat, Phi, SNR_DB, rng)

        # PositionNet input: |Phi^H y_DD|
        X_pos = np.abs((PhiH @ y_DD.T).T).astype(np.float32)

        # Binary support label: 1 where channel is non-zero
        Y_bin = (np.abs(h_flat) > 1e-9).astype(np.float32)

        # Reshape to channels-last 5D for Keras Conv3D
        all_X[start:start + bs] = X_pos.reshape(bs, M_TAU, N_NU, NT, 1)
        all_Y[start:start + bs] = Y_bin.reshape(bs, M_TAU, N_NU, NT, 1)

        elapsed = time() - t0
        print(f"  Batch {i + 1}/{n_batches} | {elapsed:.1f}s")

    # Save dataset + sensing matrix
    np.savez_compressed(
        SAVE_PATH,
        X=all_X,
        Y=all_Y,
        Phi_real=Phi.real,
        Phi_imag=Phi.imag,
    )

    size_mb = sum(a.nbytes for a in [all_X, all_Y]) / 1e6
    file_size = __import__("os").path.getsize(SAVE_PATH) / 1e6

    print(f"\nDone in {time() - t0:.1f}s")
    print(f"  Saved: {SAVE_PATH}")
    print(f"  X shape: {all_X.shape}  (|Phi^H y_DD|, channels-last)")
    print(f"  Y shape: {all_Y.shape}  (binary support mask)")
    print(f"  Raw size: {size_mb:.0f} MB  |  Compressed: {file_size:.0f} MB")
    print(f"  Non-zero fraction: {all_Y.mean():.4f}")


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import stim
import math
from itertools import combinations


def build_golay24_H() -> np.ndarray:
    """Build H = [B | I_12] for the extended binary Golay code [24,12,8]."""
    B = np.zeros((12, 12), dtype=np.uint8)
    qr_11 = {1, 3, 4, 5, 9}

    # First row/col ones (except diagonal handled later)
    for i in range(1, 12):
        B[0, i] = 1
        B[i, 0] = 1

    # Quadratic residue rule for indices 1..11
    for i in range(1, 12):
        for j in range(1, 12):
            if i == j:
                B[i, j] = 1
            elif (j - i) % 11 in qr_11:
                B[i, j] = 1

    H = np.hstack([B, np.eye(12, dtype=np.uint8)]).astype(np.uint8)
    return H


def _pack_bits_to_int(bits: np.ndarray) -> int:
    """Pack a length-m array of 0/1 bits into an integer using little-endian order."""
    # bits[0] is LSB
    m = bits.size
    out = 0
    for i in range(m):
        out |= (int(bits[i]) & 1) << i
    return out


def build_syndrome_correction_table_golay24(H: np.ndarray) -> np.ndarray:
    """
    Build a complete correction lookup table for extended Golay [24,12,8].

    Returns:
      corr_table: np.uint32 array of shape (4096,)
        corr_table[s] is a 24-bit mask to XOR with the measured data bits.

    Strategy:
      Fill syndromes with minimal-weight coset leaders by enumerating error masks
      in increasing weight order up to weight 4 (covering radius 4).
    """
    H = (H.copy() & 1).astype(np.uint8)
    m, n = H.shape  # m=12, n=24
    assert m == 12 and n == 24

    # Pack each column of H into a 12-bit integer
    col_int = np.zeros(n, dtype=np.uint16)
    for j in range(n):
        col_int[j] = _pack_bits_to_int(H[:, j])

    # Correction table indexed by 12-bit syndrome (0..4095)
    corr_table = np.zeros(1 << m, dtype=np.uint32)
    filled = np.zeros(1 << m, dtype=bool)

    # Weight 0
    filled[0] = True
    corr_table[0] = 0

    # Helper to set entry if empty
    def try_set(syn: int, mask: int):
        if not filled[syn]:
            filled[syn] = True
            corr_table[syn] = np.uint32(mask)

    # Weight 1
    for i in range(n):
        syn = int(col_int[i])
        mask = 1 << i
        try_set(syn, mask)

    # Weight 2
    for i, j in combinations(range(n), 2):
        syn = int(col_int[i] ^ col_int[j])
        mask = (1 << i) ^ (1 << j)
        try_set(syn, mask)

    # Weight 3
    for i, j, k in combinations(range(n), 3):
        syn = int(col_int[i] ^ col_int[j] ^ col_int[k])
        mask = (1 << i) ^ (1 << j) ^ (1 << k)
        try_set(syn, mask)

    # Weight 4 (covering radius for extended Golay)
    for i, j, k, l in combinations(range(n), 4):
        syn = int(col_int[i] ^ col_int[j] ^ col_int[k] ^ col_int[l])
        mask = (1 << i) ^ (1 << j) ^ (1 << k) ^ (1 << l)
        try_set(syn, mask)

    if not np.all(filled):
        missing = np.where(~filled)[0]
        raise RuntimeError(f"Correction table incomplete, missing {missing.size} syndromes.")

    return corr_table


H = build_golay24_H()
corr_table = build_syndrome_correction_table_golay24(H)

print("H shape:", H.shape)
print("HH^T mod 2 == 0 ?", bool(np.all((H @ H.T) % 2 == 0)))
print("Correction table size:", corr_table.size, "(expected 4096)")




def golay24_x_memory_circuit(H: np.ndarray, p: float) -> stim.Circuit:
    """
    One-shot X-error memory with Z-parity checks defined by H.

    Qubits:
      data: 0..23
      anc:  24..35  (12 ancillas, one per check row)

    Noise:
      X_ERROR(p) on each data qubit (independent bit-flips).

    Measurements (in order):
      12 ancilla Z measurements (syndrome bits)
      24 data Z measurements (data bits after noise, before correction)
    """
    H = (H.copy() & 1).astype(np.uint8)
    m, n = H.shape
    assert m == 12 and n == 24

    data = list(range(n))
    anc = list(range(n, n + m))  # 24..35

    c = stim.Circuit()

    # Prepare |0...0> for data and ancillas
    c.append("R", data + anc)

    # Apply bit-flip noise to data qubits
    c.append("X_ERROR", data, p)

    # Parity checks: ancilla accumulates XOR of selected data bits
    # Implemented with CX from data -> anc
    for row in range(m):
        ops = []
        a = anc[row]
        ones = np.where(H[row] == 1)[0]
        for q in ones:
            ops += [int(q), int(a)]
        if ops:
            c.append("CX", ops)

    # Measure ancillas (syndrome)
    c.append("M", anc)

    # Measure data (so decoding success can be evaluated)
    c.append("M", data)

    return c




def simulate_golay24_quantum(H: np.ndarray, corr_table: np.ndarray, p: float, shots: int, seed: int = 0) -> float:
    """
    Quantum simulation using Stim:
      - sample syndrome bits from ancilla measurements
      - sample data bits from data measurements
      - decode with corr_table (syndrome -> correction mask)
      - count block failure if residual != 0

    Returns:
      block_error_rate
    """
    H = (H.copy() & 1).astype(np.uint8)
    m, n = H.shape
    assert m == 12 and n == 24
    assert corr_table.size == 4096

    c = golay24_x_memory_circuit(H, p)
    sampler = c.compile_sampler(seed=seed)
    ms = sampler.sample(shots)  # shape: (shots, 12 + 24)

    synd = ms[:, :m].astype(np.uint16)
    data = ms[:, m:].astype(np.uint32)

    # Vectorized packing of bits into ints (little-endian)
    w12 = (1 << np.arange(m, dtype=np.uint16))
    w24 = (1 << np.arange(n, dtype=np.uint32))

    syn_int = (synd @ w12).astype(np.uint16)     # (shots,)
    data_int = (data @ w24).astype(np.uint32)    # (shots,)

    corr_int = corr_table[syn_int]               # (shots,)
    residual = data_int ^ corr_int

    ber = float(np.mean(residual != 0))
    return ber


def golay24_theory_block_fail_t3(p: float) -> float:
    """Bounded-distance t=3 line: failure probability P[W >= 4], W ~ Binomial(24, p)."""
    return 1.0 - sum(math.comb(24, w) * (p ** w) * ((1 - p) ** (24 - w)) for w in range(4))


def golay24_theory_block_fail_for_table(p: float, corr_table: np.ndarray) -> float:
    """
    Exact all-zero block failure probability for the current syndrome lookup table.

    Success happens only when the physical error equals the selected coset leader
    for its syndrome. This function evaluates that exact probability.
    """
    assert corr_table.size == 4096
    weights = ((corr_table[:, None] >> np.arange(24, dtype=np.uint32)) & 1).sum(axis=1)
    counts = np.bincount(weights.astype(np.int64), minlength=25)
    p_success = sum(float(counts[w]) * (p ** w) * ((1 - p) ** (24 - w)) for w in range(25) if counts[w])
    return 1.0 - p_success



# ----------------------------
# Parameters (tune as needed)
# ----------------------------
p_values = np.geomspace(3e-3, 2e-1, 18)
shots_L1 = 200_000
shots_L2 = 200_000
num_iters = 2
batch_size = 5_000
num_iters = 2           # number of row/col decoding sweeps
seed_base = 12345
eps = 1e-12

# -------------------------
# Configuration
# -------------------------
levels = [1, 2, 3]
eps = 1e-12

# Physical error rates (log-spaced, where Golay helps)
p_min, p_max = 1e-4, 8e-2
num_points = 24
p_values = np.geomspace(p_min, p_max, num_points)

# Shots per level (keep L2/L3 smaller; L1 is the expensive Stim part)
shots_L1 = 300_000
shots_L2 = 200_000
shots_L3 = 100_000

seed_base = 12345

# -------------------------
# Helpers
# -------------------------
def sample_level1_fail_bits_from_stim(H: np.ndarray, corr_table: np.ndarray, p: float, shots: int, seed: int) -> np.ndarray:
    """
    Run the Stim circuit for Golay(24,12,8) X-memory and return a 0/1 array:
      fail=1 if decoding does NOT return the all-zero codeword (i.e., corrected data != 0).
    """
    H = (H.copy() & 1).astype(np.uint8)
    m, n = H.shape
    assert (m, n) == (12, 24)
    assert corr_table.size == 4096

    c = golay24_x_memory_circuit(H, p)
    sampler = c.compile_sampler(seed=seed)
    ms = sampler.sample(shots)  # shape: (shots, 12 + 24), bits are 0/1

    synd = ms[:, :m].astype(np.uint16)
    data = ms[:, m:].astype(np.uint32)

    # Pack bits into ints (little-endian)
    w12 = (1 << np.arange(m, dtype=np.uint16))
    w24 = (1 << np.arange(n, dtype=np.uint32))

    syn_int = (synd @ w12).astype(np.uint16)      # (shots,)
    data_int = (data @ w24).astype(np.uint32)     # (shots,)

    corr_int = corr_table[syn_int].astype(np.uint32)
    corrected = data_int ^ corr_int               # should be 0 if fully corrected (all-zero codeword)

    fail_bits = (corrected != 0).astype(np.uint8)
    return fail_bits


def decode_outer_fail_bits(H: np.ndarray, corr_table: np.ndarray, err_bits: np.ndarray) -> np.ndarray:
    """
    Outer Golay decoder: given err_bits of shape (shots, 24) (0/1),
    compute syndrome, apply correction, and return fail bits:
      fail=1 if residual != 0 after decoding.
    """
    H = (H.copy() & 1).astype(np.uint8)
    m, n = H.shape
    assert (m, n) == (12, 24)
    assert err_bits.shape[1] == 24
    assert corr_table.size == 4096

    err_bits = err_bits.astype(np.uint8)

    # Syndrome bits and packing
    synd = (err_bits @ H.T) % 2                   # (shots, 12)
    w12 = (1 << np.arange(m, dtype=np.uint16))
    syn_int = (synd.astype(np.uint16) @ w12).astype(np.uint16)

    # Pack error bits into int
    w24 = (1 << np.arange(n, dtype=np.uint32))
    err_int = (err_bits.astype(np.uint32) @ w24).astype(np.uint32)

    corr_int = corr_table[syn_int].astype(np.uint32)
    residual = err_int ^ corr_int

    fail_bits = (residual != 0).astype(np.uint8)
    return fail_bits


##---------concat helpers ---------



# ----------------------------
# Sanity checks on inputs
# ----------------------------
H = (H.copy() & 1).astype(np.uint8)
assert H.shape == (12, 24)
assert corr_table.size == 4096

# Precompute correction bits for fast XOR:
# corr_bits[s] gives a length-24 bit-vector correction for syndrome s.
corr_bits = ((corr_table[:, None] >> np.arange(24, dtype=np.uint32)) & 1).astype(np.uint8)  # (4096, 24)
w12 = (1 << np.arange(12, dtype=np.uint16))  # weights for packing syndrome bits into an int




def golay_product_L2_circuit(p: float) -> stim.Circuit:
    """
    24x24 data qubits: indices 0..575 mapped as q(r,c)=r*24+c.
    Initialize to |0>, apply X_ERROR(p) to all data qubits, measure all.
    """
    n = 24 * 24
    c = stim.Circuit()
    data = list(range(n))
    c.append("R", data)
    c.append("X_ERROR", data, p)
    c.append("M", data)
    return c

def _decode_rows_inplace(grid: np.ndarray) -> None:
    """
    Decode each of the 24 rows as an independent Golay(24,12,8) block.
    grid shape: (B, 24, 24), uint8 bits.
    """
    B = grid.shape[0]
    rows = grid.reshape(B * 24, 24)                      # (B*24, 24)
    synd = (rows @ H.T) & 1                              # (B*24, 12) over GF(2)
    syn_int = (synd.astype(np.uint16) @ w12).astype(np.uint16)   # (B*24,)
    rows ^= corr_bits[syn_int]                           # apply correction
    grid[:] = rows.reshape(B, 24, 24)

def _decode_cols_inplace(grid: np.ndarray) -> None:
    """
    Decode each of the 24 columns as an independent Golay(24,12,8) block.
    grid shape: (B, 24, 24), uint8 bits.
    """
    B = grid.shape[0]
    gT = grid.transpose(0, 2, 1)                         # (B, 24, 24) now "rows" are original columns
    cols = gT.reshape(B * 24, 24)                        # (B*24, 24)
    synd = (cols @ H.T) & 1                              # (B*24, 12)
    syn_int = (synd.astype(np.uint16) @ w12).astype(np.uint16)
    cols ^= corr_bits[syn_int]
    grid[:] = cols.reshape(B, 24, 24).transpose(0, 2, 1)

def simulate_golay_product_L2_quantum(p: float, shots: int, seed: int = 0, iters: int = 2) -> float:
    """
    Returns block failure rate for the 24x24 Golay×Golay product code.
    Failure if residual grid != 0 after iterative row/column decoding.
    """
    c = golay_product_L2_circuit(p)
    sampler = c.compile_sampler(seed=seed)

    fails = 0
    done = 0

    while done < shots:
        B = min(batch_size, shots - done)
        ms = sampler.sample(B).astype(np.uint8)          # (B, 576) bits
        grid = ms.reshape(B, 24, 24)                     # map into 24x24

        # Iterative product-code decoding: rows then cols (repeat)
        for _ in range(iters):
            _decode_rows_inplace(grid)
            _decode_cols_inplace(grid)

        # Success if all-zero after decoding
        batch_fails = np.any(grid.reshape(B, -1), axis=1).sum()
        fails += int(batch_fails)
        done += B

    return fails / shots


def _decode_axis_inplace_3d(cube: np.ndarray, axis: int) -> None:
    """
    Decode all length-24 lines of a 24x24x24 cube along one axis.
    cube shape: (B, 24, 24, 24), axis in {1, 2, 3} (batch axis is 0).
    """
    assert cube.ndim == 4 and cube.shape[1:] == (24, 24, 24)
    assert axis in (1, 2, 3)

    # Move target axis to the last position, decode all lines in a single batched pass.
    view = np.moveaxis(cube, axis, -1)                   # (B, 24, 24, 24)
    lines = view.reshape(-1, 24)                         # (B*24*24, 24)
    synd = (lines @ H.T) & 1                             # (B*24*24, 12)
    syn_int = (synd.astype(np.uint16) @ w12).astype(np.uint16)
    lines ^= corr_bits[syn_int]
    view[:] = lines.reshape(view.shape)


def golay_product_L3_circuit(p: float) -> stim.Circuit:
    """
    24x24x24 data qubits (n=24^3). Prepare |0>, apply X_ERROR(p), then measure all.
    """
    n = 24 * 24 * 24
    c = stim.Circuit()
    data = list(range(n))
    c.append("R", data)
    c.append("X_ERROR", data, p)
    c.append("M", data)
    return c


def simulate_golay_product_L3_quantum(
    p: float,
    shots: int,
    seed: int = 0,
    iters: int = 2,
    batch_size_l3: int = 128,
) -> float:
    """
    True L=3 simulation for Golay^3 on a 24x24x24 grid.

    Decoding applies iterative line decoding along x, y, z axes. A shot fails if
    any residual bit is non-zero after decoding.
    """
    c = golay_product_L3_circuit(p)
    sampler = c.compile_sampler(seed=seed)

    fails = 0
    done = 0

    while done < shots:
        B = min(batch_size_l3, shots - done)
        ms = sampler.sample(B).astype(np.uint8)          # (B, 13824) bits
        cube = ms.reshape(B, 24, 24, 24)

        for _ in range(iters):
            _decode_axis_inplace_3d(cube, axis=1)
            _decode_axis_inplace_3d(cube, axis=2)
            _decode_axis_inplace_3d(cube, axis=3)

        batch_fails = np.any(cube.reshape(B, -1), axis=1).sum()
        fails += int(batch_fails)
        done += B

    return fails / shots


def estimate_curve_crossing(
    x: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    tol: float = 1e-15,
) -> float | None:
    """
    Estimate first crossing x where y_a(x) == y_b(x) by linear interpolation.
    Returns None if no crossing is found on the sampled interval.
    """
    x = np.asarray(x, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    d = y_a - y_b
    d[np.abs(d) <= tol] = 0.0

    assert x.ndim == y_a.ndim == y_b.ndim == 1
    assert x.size == y_a.size == y_b.size
    assert x.size >= 2

    # Prefer strict sign changes between adjacent sampled points.
    for i in range(x.size - 1):
        d0 = d[i]
        d1 = d[i + 1]
        if d0 * d1 < 0:
            t = d0 / (d0 - d1)
            return float(x[i] + t * (x[i + 1] - x[i]))

    # If no strict change, accept exact-zero points only when they are
    # bracketed by opposite non-zero signs (real crossing landing on a sample).
    zero_idx = np.where(d == 0)[0]
    for i in zero_idx:
        left = d[:i][d[:i] != 0]
        right = d[i + 1:][d[i + 1:] != 0]
        if left.size and right.size and left[-1] * right[0] < 0:
            return float(x[i])

    return None


def estimate_physical_thresholds(
    p_values: np.ndarray,
    pL1: np.ndarray,
    pL2: np.ndarray,
    pL3: np.ndarray,
) -> dict:
    """
    Return practical threshold-like crossing estimates from sampled logical curves.
    """
    p_values = np.asarray(p_values, dtype=float)
    pL1 = np.asarray(pL1, dtype=float)
    pL2 = np.asarray(pL2, dtype=float)
    pL3 = np.asarray(pL3, dtype=float)

    return {
        "pseudo_L1": estimate_curve_crossing(p_values, pL1, p_values),
        "pseudo_L2": estimate_curve_crossing(p_values, pL2, p_values),
        "pseudo_L3": estimate_curve_crossing(p_values, pL3, p_values),
        "cross_L1_L2": estimate_curve_crossing(p_values, pL1, pL2),
        "cross_L2_L3": estimate_curve_crossing(p_values, pL2, pL3),
    }

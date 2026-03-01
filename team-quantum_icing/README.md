# Team Quantum Icing — QPFL Hackathon 2026

Fault-tolerant quantum computation on **Cat Qubits** using the Steane `[7,1,3]`, Binary Golay `[23,12,7]`, extended Binary Golay `[24,12,8]`, and LDPC codes, simulated end-to-end with **Stim** and PyMatching.

---

## Repository Structure

```
team-quantum_icing/
├── Steane.ipynb          # Steane [7,1,3] experiments
├── Binary_Golay.ipynb    # Extended Golay [24,12,8] experiments
├── golay24.py            # Reusable Golay-24 Python module
├── golay23.py            # Binary Golay [23,12,7] — circuit, decoder, BER simulation
├── core/
│   ├── LDPC.ipynb        # LDPC code exploration (WIP)
│   ├── LDPC_homemade.ipynb
│   └── LDPC_stim.ipynb
└── README.md
```

---

## Steane.ipynb

Explores fault-tolerant operations on a **Hamming/Steane `[7,1,3]`** encoded Cat Qubit. Beyond basic error correction, we derive and simulate the full set of **logical Clifford gates** adapted to the Steane code architecture, analysing which operations are natively safe for Cat Qubits and which require Gate Teleportation to preserve the hardware bias.

### 1 · Hamming X-Memory Baseline
- Builds a one-shot X-error memory circuit (7 data qubits + 3 ancillas).
- Z-parity checks via `CZ`–`H` syndrome extraction; logical observable defined as a parity of data measurements.
- Decodes with **PyMatching**; threshold confirmed at ~7 × 10⁻³ block failure rate at p = 1%.

### 2 · Transversal CNOT — Native Fault-Tolerance
Implements a transversal CNOT between two 7-qubit encoded blocks (14 data + 6 ancilla qubits).

| Scenario | Setup | Result |
|---|---|---|
| A – Deterministic | Single injected X on control qubit 2 | **0.00% logical failure** — both decoders correct one error each |
| B – Stochastic | i.i.d. X_ERROR(p=1%) on all 14 data qubits | **~7.5 × 10⁻³** — noise stays within code distance |

**Key finding:** Transversal CNOT is bias-preserving (X → X, Z → Z), making it the safe native gate for Cat Qubits.

### 3 · Hadamard Trap — Why Transversal H Destroys Bias
| Scenario | Setup | Z-error rate |
|---|---|---|
| A – Transversal H | X noise + physical H on each qubit | **~51.87%** — X errors rotate to Z, bias destroyed |
| B – Gate Teleportation | X noise + logical H via transversal CNOT routing | **0.00%** — bias perfectly preserved |

### 4 · S-Gate Anomaly — Why Phase Gates Also Break Cat Qubits
- Transversal S conjugates X → Y, introducing simultaneous X and Z components.
- **100% syndrome matching** between X-stabilizer and Z-stabilizer triggers proves every X error is converted to a Y error.
- Conclusion: both H and S gates must be executed exclusively via Gate Teleportation.

---

## Binary_Golay.ipynb

Extends the same architectural analysis to the **extended Binary Golay `[24,12,8]`** code.

### 1 · Code Construction & Syndrome Table
- Constructs H = [B | I₁₂] using the quadratic-residue rule over GF(2).
- Precomputes a complete 12-bit → correction-mask lookup table by exhaustively enumerating all error patterns of weight ≤ 4, guaranteeing unique decoding for every possible syndrome.

### 2 · X-Memory Quantum Simulation
- Stim circuit: 24 data qubits + 12 ancillas, Z-parity checks via CX syndrome extraction.
- Plots block failure rate versus physical bit-flip probability p and compares to the simple theoretical ≥4-flip bound.

### 3 · Hierarchical Concatenation (L = 1, 2, 3)
- L=1: direct Stim simulation.
- L=2/3: outer Golay decoder applied to 24 independent L-1 logical failure bits (bootstrap Monte Carlo).
- Confirms super-threshold separation and logical error suppression with concatenation depth.

### 4 · True L=2 Product Code (24 × 24 = 576 qubits)
- Stim circuit generates X errors on a 576-qubit grid; decoded by iterative row/column Golay sweeps.
- Compared against bootstrapped L=2 to validate the product-code architecture.

### CNOTs — bias-preserving routing

Transversal CNOTs preserve the bias (X → X, Z → Z) and were used as routing primitives for Gate Teleportation. We verified in Golay simulations that CNOT-based routing moves logical states between blocks without introducing new Z-type errors at both 24- and 576-qubit scales.

### 5 · Hadamard & S-Gate Traps at Scale (Syndrome Differencing)
Uses **syndrome differencing** (XOR final syndrome with baseline) to avoid post-selection on 24-qubit state preparation.

| Gate | X-syndrome | Z-syndrome | Conclusion |
|---|---|---|---|
| None (control, p=5%) | ~71% | **0%** | Pure Cat Qubit noise |
| Transversal H | ~0% | **71%** | Complete axis swap |
| Transversal S | ~71% | **71%** (100% match) | X → Y, lethal Z introduced |

**Architectural verdict:** All axis-mixing gates (H, S, T) must be implemented via Gate Teleportation using bias-preserving transversal CNOTs.

---

## golay23 — Binary Golay

Standalone Python module for the perfect binary Golay code (n=23, k=12, d=7, t=3).

### 1 · Code Construction
- Derives the systematic generator matrix G (12×23) from the cyclic generator polynomial `g(x) = 1+x+x⁵+x⁶+x⁷+x⁹+x¹¹` via GF(2) RREF.
- Builds the parity-check matrix H (11×23) as `H = [Pᵀ | I₁₁]` and verifies `G·Hᵀ = 0` over GF(2).

### 2 · Syndrome Correction Table
- Enumerates all error patterns of weight ≤ 3, mapping each of the 2048 syndromes to its minimum-weight coset leader.
- Guarantees complete unique decoding for all correctable errors (t = 3).

### 3 · Quantum X-Memory Circuit
- Stim circuit: 23 data qubits + 11 ancillas; Z-parity checks via CX syndrome extraction followed by data measurement.
- Decodes each shot by syndrome lookup and corrects the 23 measured bits.

### 4 · Block Error Rate
- Sweeps physical bit-flip probability p and computes the simulated BER.
- Compares against the exact theoretical bound for BSC(p) with t=3 correction: `P(W ≥ 4)`.

---

## LDPC (core)

The `core/` folder contains three notebooks focused on LDPC codes and practical decoders:

- `LDPC.ipynb`: construction and analysis of parity-check matrices (PEG, regular/irregular), Tanner graph properties (girth, degree) and GF(2) RREF.
- `LDPC_homemade.ipynb`: implementations and comparisons of homemade decoders (sum-product, min-sum, bit-flipping), convergence analysis and software/hardware optimizations.
- `LDPC_stim.ipynb`: integration with Stim for hybrid simulations (error→measurement mapping, sampling, BER/FER evaluation) and direct comparisons between classical decoders and Stim-based workflows.

The goal is to evaluate scalable LDPC constructions and efficient decoders, and to compare their performance to the short-block codes used elsewhere in this project.

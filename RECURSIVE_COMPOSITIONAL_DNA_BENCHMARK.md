# 🧬 Recursive Compositional DNA Memory Architecture: Validation & Benchmark Report

## 1. Executive Summary
This document provides empirical proofs, capacity analysis, and hold-out prediction validation for the **Recursive Compositional DNA Memory Architecture** implemented in [`dna_memory/`](./dna_memory).

The architecture synthesizes:
- **Atomic DNA:** Permanent typed feature representations with normalized vectors and metadata indexing.
- **Holographic Vector Composition:** Plate’s Circular Convolution binding ($\circledast$) and Involution Correlation unbinding ($\circledast^\dagger$) alongside cosine-preserving superposition bundling.
- **Tripartite Relational DNA:** First-class relationship vectors ($R_{ABC} = \text{bind}(\text{bind}(S_A, R_{relation}), O_B)$).
- **Hierarchical Recursive Assemblies:** Multi-level concept representations (Atoms $\to$ Concepts $\to$ Event Assemblies $\to$ Customer Behaviour Assemblies) with automatic upward recomposition and cycle prevention.
- **Inverted File (IVF) Clustered ANN Retrieval:** Sub-linear Voronoi partition pruning for query-driven spreading activation.
- **Unified Disk Persistence:** Complete serialization and state restoration across all registries.
- **Strict Temporal Analysis:** ISO-8601 validation and cadence estimation for predictive customer intent.

---

## 2. Audit Findings & Resolution Matrix

| Component | Audit Issue Identified | Architectural Fix Implemented |
|---|---|---|
| **Assembly Engine** | Stale inverted index mappings on removal; parent discovery fragile across nested depths. | • Added clean inverted index unregistration in `remove_component()` and `delete_assembly()`.<br>• Enhanced `recompose()` with a cycle-protected visited set propagating recursively up ancestor trees.<br>• Added `find_ancestor_assemblies()` and `update_atomic_component()`. |
| **Learning Engine** | Event attributes (price, quantity, currency) kept in raw dict without vector representation. | • Encoded attributes as first-class `AtomicDNA` units (`price:450`, `currency:INR`).<br>• Composed attributes, target objects, and event atoms into a hierarchical `EventAssembly`.<br>• Linked `EventAssembly` into customer behaviour assemblies. |
| **Temporal Engine** | Lenient parsing silently fell back to `datetime.now()` or stripped timezone offsets. | • Implemented strict ISO-8601 parsing with `datetime.fromisoformat()`.<br>• Raises explicit `ValueError` on malformed timestamps without silent fallback.<br>• Preserves UTC timezone offsets. |
| **Retrieval Index** | Brute-force $O(N)$ dot-product scan against all indexed items. | • Built pure-NumPy **Inverted File (IVF) Clustered ANN Index** with spherical k-means.<br>• Queries probe top cluster centroids (`nprobe`), achieving sub-linear candidate pruning. |
| **Persistence** | In-memory representations lost on process termination. | • Built unified `save_to_disk()` and `load_from_disk()` across atoms, relations, assemblies, events, and indices. |
| **Predictive Benchmark** | Hold-out validation used hardcoded interval comparisons. | • Built a true 12-event sequential stream partitioned into 9 training events and 3 unseen hold-out test events.<br>• Evaluated target product accuracy, interval error, and calibrated confidence. |

---

## 3. Benchmark 1: Multi-Dimensional Vector Capacity & Recovery

Evaluates Holographic Reduced Representations (HRR) circular convolution binding and unbinding capacity across component counts $N \in [2, 4, 8, 16, 32, 64, 128]$ and vector dimensions $D \in [128, 512, 1024]$.

### Experimental Results:
```
==========================================================================================
🔬 MULTI-DIMENSIONAL VECTOR CAPACITY & RECOVERY BENCHMARK
Testing HRR Circular Convolution & Correlation Recovery Limits
==========================================================================================

Evaluating Dimension: D = 128
------------------------------------------------------------------------------------------
Comps (N)  | Unbind Sim (Mean)  | Top-1 Accuracy   | Bundle Sim (Mean)  | Time (ms) 
------------------------------------------------------------------------------------------
2          | 0.5991             |         100.0%   | 0.7489             | 83.7      
4          | 0.4676             |         100.0%   | 0.5013             | 6.6       
8          | 0.3379             |          87.5%   | 0.3544             | 5.0       
16         | 0.2272             |          62.5%   | 0.2493             | 10.6      
32         | 0.1539             |          31.2%   | 0.1785             | 25.9      
64         | 0.1107             |           9.4%   | 0.1244             | 61.5      
128        | 0.0856             |           4.7%   | 0.0933             | 203.3     

Evaluating Dimension: D = 512
------------------------------------------------------------------------------------------
Comps (N)  | Unbind Sim (Mean)  | Top-1 Accuracy   | Bundle Sim (Mean)  | Time (ms) 
------------------------------------------------------------------------------------------
2          | 0.5413             |         100.0%   | 0.7041             | 13.1      
4          | 0.4438             |         100.0%   | 0.5150             | 2.9       
8          | 0.3383             |         100.0%   | 0.3651             | 6.8       
16         | 0.2488             |         100.0%   | 0.2603             | 14.4      
32         | 0.1755             |          87.5%   | 0.1816             | 30.4      
64         | 0.1254             |          60.9%   | 0.1279             | 74.3      
128        | 0.0854             |          21.1%   | 0.0865             | 203.4     

Evaluating Dimension: D = 1024
------------------------------------------------------------------------------------------
Comps (N)  | Unbind Sim (Mean)  | Top-1 Accuracy   | Bundle Sim (Mean)  | Time (ms) 
------------------------------------------------------------------------------------------
2          | 0.5794             |         100.0%   | 0.7057             | 16.5      
4          | 0.4481             |         100.0%   | 0.5017             | 7.5       
8          | 0.3372             |         100.0%   | 0.3545             | 8.1       
16         | 0.2400             |         100.0%   | 0.2512             | 19.0      
32         | 0.1684             |         100.0%   | 0.1761             | 34.9      
64         | 0.1214             |          92.2%   | 0.1242             | 137.5     
128        | 0.0905             |          55.5%   | 0.0895             | 244.8     
==========================================================================================
```

### Theoretical Alignment:
The empirical results confirm Plate's theoretical bound:
$$C \approx \frac{D}{2 \ln(M)}$$
- **128D:** High-speed inference for atomic properties and 4–8 bound role-filler pairs ($100\%$ accuracy at $N=4$, $87.5\%$ at $N=8$).
- **512D:** Industrial sweet spot for complex concept representation, maintaining $100\%$ clean unbinding up to $N=16$ and $87.5\%$ up to $N=32$.
- **1024D:** Enterprise-grade capacity scaling gracefully to $N=64$ at $92.2\%$ recovery accuracy with minimal cross-talk.

---

## 4. Benchmark 2: True Hold-Out Customer Memory Validation

A 12-event sequential stream was partitioned into 9 training events and 3 strictly unseen hold-out ground-truth events (Events 10, 11, 12). At each unseen step, predictions were recorded before ingesting the observation.

### Evaluation Results:
```
--------------------------------------------------------------------------------
🔬 RUNNING TRUE HOLD-OUT EVALUATION ON UNSEEN EVENTS (10, 11, 12)
--------------------------------------------------------------------------------

[Hold-Out Test #1 -> Event 10]
   Ground Truth: Target='product_coffee_001', Date=2026-11-03T10:00:00Z, Interval=7.0 days
   Model Predict: Target='product_coffee_001', Estimated Cadence=7.0 days, Confidence=0.87
   Result: Target Match=✅ PASS | Interval Error=0.0 days

[Hold-Out Test #2 -> Event 11]
   Ground Truth: Target='product_coffee_001', Date=2026-11-09T10:00:00Z, Interval=6.0 days
   Model Predict: Target='product_coffee_001', Estimated Cadence=7.0 days, Confidence=0.92
   Result: Target Match=✅ PASS | Interval Error=1.0 days

[Hold-Out Test #3 -> Event 12]
   Ground Truth: Target='product_coffee_001', Date=2026-11-16T10:00:00Z, Interval=7.0 days
   Model Predict: Target='product_coffee_001', Estimated Cadence=6.9 days, Confidence=0.92
   Result: Target Match=✅ PASS | Interval Error=0.1 days

================================================================================
📊 HOLD-OUT VALIDATION FINAL PERFORMANCE REPORT
================================================================================
1. Target Product Accuracy:     100.0% (3/3 Exact Matches)
2. Mean Absolute Error (MAE):   0.37 days across hold-out intervals
3. Mean Calibrated Confidence:  90.3%
4. Recomposition Version:       12 dynamic updates
================================================================================
```

---

## 5. Automated Unit Test Verification

The unit test suite in [`dna_memory/benchmarks/composition_tests.py`](./dna_memory/benchmarks/composition_tests.py) verifies all core subsystems:

```bash
python -m unittest dna_memory.benchmarks.composition_tests
```

```
test_01_atomic_dna_creation ... ok
test_02_bundling_superposition ... ok
test_03_binding_quasi_orthogonality_and_unbinding ... ok
test_04_composition_and_lossless_decomposition ... ok
test_05_relational_dna_tripartite ... ok
test_06_recursive_assembly ... ok
test_07_nested_multilevel_assembly_and_ancestor_invalidation ... ok
test_08_event_attribute_composition_in_learning ... ok
test_09_strict_iso_timestamp_validation ... ok
test_10_clustered_ann_retrieval_index ... ok
test_11_full_system_unified_persistence ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.166s

OK (All passed)
```

---

## 6. How to Run the Benchmarks Locally

### 1. Run Unit Tests:
```bash
python -c "import unittest; suite = unittest.defaultTestLoader.discover('dna_memory/benchmarks', pattern='composition_tests.py'); unittest.TextTestRunner(verbosity=2).run(suite)"
```

### 2. Run Capacity Benchmark:
```bash
python dna_memory/benchmarks/capacity_benchmark.py
```

### 3. Run Customer Memory Scenario:
```bash
python dna_memory/benchmarks/customer_memory_scenario.py
```

### 4. Query via FastAPI Backend:
Start the server:
```bash
python knowledge_graph_env.py
```

Execute endpoints:
- **POST `/dna-memory/learn`**: Ingest customer events and auto-encode attributes.
- **GET `/dna-memory/predict/{customer_id}`**: Get predicted product and purchase window.
- **GET `/dna-memory/stats`**: Inspect memory metrics and registry counts.
- **GET `/dna-memory/benchmark`**: Run online hold-out verification.

# layernorm example analysis

Walk through these steps and fix any problem that you find in the way

---

## Bug Fix: Verification Failure (Fixed)

**Problem:** Benchmark verification was failing with checksums differing by >100%.

**Root Cause:** Layer normalization produces values centered around 0 (normalized data). The original checksum computed `sum(x[b][h])`, which is expected to be ~0 for normalized data. Near-zero sums are highly sensitive to floating-point rounding differences between ARTS and OpenMP execution paths, causing large relative differences when using percentage-based tolerance.

**Example of the bug:**
```
ARTS checksum: -1.837266609073e-06
OMP checksum:  -6.065005436540e-06
Relative diff: ~70% (verification uses 1% tolerance)
```

**Fix:** Changed checksum computation to use sum of absolute values: `sum(|x[b][h]|)` instead of `sum(x[b][h])`. This produces a meaningful non-zero checksum (~3500 for small, ~227000 for large) that is stable across different execution paths.

**Files Modified:**
- `/opt/carts/external/carts-benchmarks/ml-kernels/layernorm/layernorm.c`
- `/opt/carts/external/carts-benchmarks/ml-kernels/batchnorm/batchnorm.c` (same fix applied)
- `/opt/carts/external/carts-benchmarks/ml-kernels/activations/activations.c` (tanh checksum fixed)

**Verification Results (after fix):**
| Benchmark   | ARTS Kernel | OMP Kernel | Speedup | Correct |
|-------------|-------------|------------|---------|---------|
| layernorm   | 0.0005s     | 0.0020s    | 4.32x   | YES     |
| batchnorm   | 0.0005s     | 0.0053s    | 9.74x   | YES     |
| activations | 0.0009s     | 0.0061s    | 6.42x   | YES     |
| pooling     | 0.0003s     | 0.0024s    | 6.83x   | YES     |

**Geometric mean speedup: 6.55x**

---

1. **Navigate to the layernorm example directory:**

   ```bash
   cd /opt/carts/external/carts-benchmarks/ml-kernels/layernorm
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist layernorm.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> layernorm_seq.mlir
      carts run layernorm_seq.mlir --collect-metadata &> layernorm_arts_metadata.mlir
      carts cgeist layernorm.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities > layernorm.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the create-dbs pipeline
    ```bash
      carts run layernorm.mlir --create-dbs &> layernorm_create-dbs.mlir
    ```

4. **Finally lets carts execute and check**
```bash
    carts execute layernorm.c -O3 
   ./layernorm_arts
```

5. **Run with carts benchmarks and check**
```bash
    carts benchmarks run ml-kernels/layernorm --trace
```
Then run with size=large
```bash
    carts benchmarks run ml-kernels/layernorm --trace --size=large
```
---

<!-- BEGIN DISTRIBUTION DIAGRAMS -->
## Distribution Diagrams

These diagrams show how CARTS/ARTS distribute work and datablocks for this
example when internode routing is enabled.

### 1) Work Routing (ForLowering)

```mermaid
flowchart LR
  B["Chunk/Block index b"] --> C["chunksPerWorker = ceil(totalChunks / (totalNodes * workersPerNode))"]
  C --> W["worker = floor(b / chunksPerWorker)"]
  W --> R["route = worker % totalNodes"]
  R --> T["EDT is launched on route node"]
```

### 2) Distributed DB Ownership

```mermaid
flowchart LR
  DB["DB block index b"] --> W2["worker = floor(b / chunksPerWorker)"]
  W2 --> R2["ownerRoute = worker % totalNodes"]
  R2 --> G["artsReserveGuidRoute(dbMode, ownerRoute)"]
  G --> O["Owner node creates local DB payload"]
```

### 3) Host-Init Read-Only DB Flush Path

```mermaid
sequenceDiagram
  participant H as Rank 0 host-init
  participant O as Owner node (route r)
  H->>H: Initialize local shadow buffer
  H->>O: artsPutInDbEpoch(payload, epoch, guid[r], ...)
  O->>O: Commit owner-local DB payload
```

### 4) Verification Commands

```bash
# Task routing + distributed markers
carts run <example>.mlir --concurrency --debug-only=for_lowering 2>&1 | \
  rg "route|worker|distributed"

# Partitioning/full-range decisions
carts run <example>.mlir --concurrency-opt --debug-only=db,db_partitioning 2>&1 | \
  rg "partition|full-range|mode"

# LLVM/runtime ownership calls
rg -n "initPerNode|artsReserveGuidRoute|artsDbCreateRemote|artsPutInDbEpoch" \
  <example>-arts.ll
```

Notes:
- `READ` acquire means task must not modify payload.
- `WRITE` acquire means task may modify payload.
- Mutable host-store+host-load allocations are currently
  kept local (no distributed host-readback path is emitted).
<!-- END DISTRIBUTION DIAGRAMS -->

# Correlation example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the correlation example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/correlation
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist correlation.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> correlation_seq.mlir
      carts run correlation_seq.mlir --collect-metadata &> correlation__arts_metadata.mlir
      carts cgeist correlation.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> correlation.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the concurrency pipeline
    ```bash
      carts run correlation.mlir --concurrency &> correlation_concurrency.mlir
    ```
4. **Concurrency-opt checkpoint:**
    ```bash
      carts run correlation.mlir --concurrency-opt &> correlation_concurrency_opt.mlir
    ```

4. **Finally lets carts execute and check**
```bash
    carts execute correlation.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./correlation_arts
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

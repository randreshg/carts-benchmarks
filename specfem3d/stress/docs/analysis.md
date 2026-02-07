# stress_update example analysis

Walk through these steps to debug CARTS pipeline issues.

## 1. Navigate to the stress example directory

```bash
cd ~/Documents/carts/external/carts-benchmarks/specfem3d/stress
```

## 2. Build CARTS if any changes were made

```bash
carts build
```

## 3. Generate MLIR from C source

Generate the sequential MLIR (for metadata):
```bash
carts cgeist stress_update.c -O0 --print-debug-info -S --raise-scf-to-affine -I../common 2>/dev/null > /tmp/stress_seq.mlir
```

Generate the parallel MLIR (with OpenMP):
```bash
carts cgeist stress_update.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I../common 2>/dev/null > /tmp/stress_par.mlir
```

## 4. Run pipeline stages incrementally

### Stage 1: collect-metadata (on sequential code)
```bash
carts run /tmp/stress_seq.mlir --collect-metadata 2>&1 | head -20
```

### Stage 2: openmp-to-arts
```bash
carts run /tmp/stress_par.mlir --openmp-to-arts > /tmp/stress_arts.mlir 2>&1
```

### Stage 3: create-dbs
```bash
carts run /tmp/stress_par.mlir --create-dbs > /tmp/stress_dbs.mlir 2>&1
```

### Stage 4: concurrency
```bash
carts run /tmp/stress_par.mlir --concurrency > /tmp/stress_conc.mlir 2>&1
```

## 5. Build and run the example

```bash
make clean && make all
./stress_update_arts
```

---

## Issues Fixed

### Issue 1: Switch statement not supported

**Original code:**
```c
static inline float derivative(const float ***arr, int i, int j, int k, int dir) {
  switch (dir) {
  case 0:
    return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
  case 1:
    return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
  default:
    return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
  }
}
```

**Problem:** The `switch` statement creates multiple basic blocks (`cf.switch` in MLIR). When inlined into a loop body, this violates MLIR's `scf.for` requirement that loop bodies have only 0 or 1 blocks.

**Error message:**
```
'scf.for' op expects region #0 to have 0 or 1 blocks
```

**Fix:** Replace the switch with separate inline functions:
```c
static inline float derivative_x(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
}

static inline float derivative_y(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
}

static inline float derivative_z(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
}
```

### Issue 2: Collapse clause not supported

**Original code:**
```c
#pragma omp parallel for collapse(2) schedule(static)
```

**Problem:** CARTS does not currently support the `collapse` clause.

**Fix:** Remove the collapse clause:
```c
#pragma omp parallel for schedule(static)
```

### Issue 3: Early return not supported

**Original code:**
```c
if (!vx || !vy || !vz || !rho || !mu || !lambda || !sxx || !syy || !szz ||
    !sxy || !sxz || !syz) {
  fprintf(stderr, "allocation failure\n");
  return 1;
}
```

**Problem:** CARTS does not support early returns that create multiple exit points from functions.

**Fix:** Remove the early return check (assume allocations succeed in this benchmark context).

### Issue 4: Missing arts.cfg

**Problem:** The ARTS runtime requires a configuration file.

**Fix:** Copy arts.cfg from another example:
```bash
cp ../common/arts.cfg .
# or from volume-integral
```

### Issue 5: Makefile naming mismatch

**Original:**
```makefile
EXAMPLE_NAME := stress
SRC := stress_update.c
```

**Problem:** EXAMPLE_NAME must match the source file basename for the pipeline to find intermediate files.

**Fix:**
```makefile
EXAMPLE_NAME := stress_update
SRC := stress_update.c
```

---

## CARTS Limitations Summary

This example demonstrates several CARTS limitations:

| Feature | Supported | Workaround |
|---------|-----------|------------|
| `switch` statements in loops | No | Use separate functions or if-else |
| `collapse(N)` clause | No | Remove collapse, use single loop parallelization |
| Early returns | No | Remove or restructure control flow |
| Multiple function exits | No | Single return point only |

---

## Verification Commands

After fixes:

```bash
# 1. Test the full build
cd ~/Documents/carts/external/carts-benchmarks/specfem3d/stress
make clean && make all

# 2. Run the executable
./stress_update_arts

# Expected output:
# specfem3d_stress checksum=0.000558
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

# rhs4sg_base example analysis

Walk through these steps to debug CARTS pipeline issues.

## 1. Navigate to the rhs4sg-base example directory

```bash
cd ~/Documents/carts/external/carts-benchmarks/sw4lite/rhs4sg-base
```

## 2. Build CARTS if any changes were made

```bash
carts build
```

## 3. Generate MLIR from C source

Generate the sequential MLIR (for metadata):
```bash
carts cgeist rhs4sg_base.c -O0 --print-debug-info -S --raise-scf-to-affine -I../common 2>/dev/null > /tmp/rhs4sg_seq.mlir
```

Generate the parallel MLIR (with OpenMP):
```bash
carts cgeist rhs4sg_base.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I../common 2>/dev/null > /tmp/rhs4sg_par.mlir
```

## 4. Run pipeline stages incrementally

### Stage 1: collect-metadata (on sequential code)
```bash
carts run /tmp/rhs4sg_seq.mlir --collect-metadata 2>&1 | head -20
```

### Stage 2: openmp-to-arts
```bash
carts run /tmp/rhs4sg_par.mlir --openmp-to-arts > /tmp/rhs4sg_arts.mlir 2>&1
```

### Stage 3: create-dbs
```bash
carts run /tmp/rhs4sg_par.mlir --create-dbs > /tmp/rhs4sg_dbs.mlir 2>&1
```

### Stage 4: concurrency
```bash
carts run /tmp/rhs4sg_par.mlir --concurrency > /tmp/rhs4sg_conc.mlir 2>&1
```

## 5. Build and run the example

```bash
make clean && make all
./rhs4sg_base_arts
```

---

## Issues Fixed

### Issue 1: Collapse clause not supported

**Original:**
```c
#pragma omp parallel for collapse(2) schedule(static)
```

**Fix:**
```c
#pragma omp parallel for schedule(static)
```

### Issue 2: If-else chain inside loop

**Original code:**
```c
for (int c = 0; c < COMP; ++c) {
  float lap = 0.0f;
  for (int offset = -2; offset <= 2; ++offset) {
    lap += w[offset + 2] * (u[c][i + offset][j][k] +
                            u[c][i][j + offset][k] +
                            u[c][i][j][k + offset]);
  }

  float div_term = 0.0f;
  if (c == 0) {
    div_term = u[c][i + 1][j][k] - u[c][i - 1][j][k];
  } else if (c == 1) {
    div_term = u[c][i][j + 1][k] - u[c][i][j - 1][k];
  } else {
    div_term = u[c][i][j][k + 1] - u[c][i][j][k - 1];
  }

  rhs[c][i][j][k] = mu_c * lap * inv_h2 + (la_c + mu_c) * div_term * (0.5f / h);
}
```

**Problem:** The `if-else if-else` chain creates multiple basic blocks in MLIR, similar to a switch statement. When inlined into a loop body, this violates `scf.for`'s requirement for 0 or 1 blocks.

**Fix:** Manually unroll the component loop (COMP=3) to avoid the conditional:
```c
// Component 0 (x-direction divergence)
{
  float lap0 = 0.0f;
  for (int offset = -2; offset <= 2; ++offset) {
    lap0 += w[offset + 2] * (u[0][i + offset][j][k] +
                             u[0][i][j + offset][k] +
                             u[0][i][j][k + offset]);
  }
  float div_term0 = u[0][i + 1][j][k] - u[0][i - 1][j][k];
  rhs[0][i][j][k] = mu_c * lap0 * inv_h2 + (la_c + mu_c) * div_term0 * (0.5f / h);
}

// Component 1 (y-direction divergence)
{
  float lap1 = 0.0f;
  for (int offset = -2; offset <= 2; ++offset) {
    lap1 += w[offset + 2] * (u[1][i + offset][j][k] +
                             u[1][i][j + offset][k] +
                             u[1][i][j][k + offset]);
  }
  float div_term1 = u[1][i][j + 1][k] - u[1][i][j - 1][k];
  rhs[1][i][j][k] = mu_c * lap1 * inv_h2 + (la_c + mu_c) * div_term1 * (0.5f / h);
}

// Component 2 (z-direction divergence)
{
  float lap2 = 0.0f;
  for (int offset = -2; offset <= 2; ++offset) {
    lap2 += w[offset + 2] * (u[2][i + offset][j][k] +
                             u[2][i][j + offset][k] +
                             u[2][i][j][k + offset]);
  }
  float div_term2 = u[2][i][j][k + 1] - u[2][i][j][k - 1];
  rhs[2][i][j][k] = mu_c * lap2 * inv_h2 + (la_c + mu_c) * div_term2 * (0.5f / h);
}
```

### Issue 3: Early return not supported

**Original:**
```c
if (!u || !mu || !lambda || !rhs) {
  fprintf(stderr, "allocation failure\n");
  return 1;
}
```

**Fix:** Remove the early return check.

### Issue 4: Makefile naming mismatch

**Original:**
```makefile
EXAMPLE_NAME := rhs4sg-base
```

**Fix:**
```makefile
EXAMPLE_NAME := rhs4sg_base
```

---

## Verification

```bash
cd ~/Documents/carts/external/carts-benchmarks/sw4lite/rhs4sg-base
make clean && make all
./rhs4sg_base_arts

# Expected output:
# sw4lite_rhs4sg_base checksum=0.019759
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

# volume-integral example analysis

Walk through these steps to debug CARTS pipeline issues and understand the SSA dominance bug that was fixed.

## 1. Navigate to the volume-integral example directory

```bash
cd ~/Documents/carts/external/carts-benchmarks/seissol/volume-integral
```

## 2. Build CARTS if any changes were made

```bash
carts build
```

## 3. Generate MLIR from C source

Generate the sequential MLIR (for metadata):
```bash
carts cgeist volume_integral.c -O0 --print-debug-info -S --raise-scf-to-affine -I../common > /tmp/volume_integral_seq.mlir
```

Generate the parallel MLIR (with OpenMP):
```bash
carts cgeist volume_integral.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I../common > /tmp/volume_integral.mlir
```

## 4. Run pipeline stages incrementally

### Stage 1: openmp-to-arts
```bash
carts run /tmp/volume_integral.mlir --openmp-to-arts > /tmp/volume_integral_arts.mlir 2>&1
```

Check the `arts.for` structure - the buffer allocation should be INSIDE the loop:
```bash
grep -A20 "arts.for" /tmp/volume_integral_arts.mlir | head -30
```

Expected output shows `memref.alloca` inside `arts.for`:
```mlir
"arts.for"(%c0, %c64, %c1) ({
^bb0(%arg0: index):
  %alloca = memref.alloca() : memref<36xf32>  // buffer[N_QUAD] - INSIDE loop
  ...
```

### Stage 2: create-dbs
```bash
carts run /tmp/volume_integral.mlir --create-dbs > /tmp/volume_integral_dbs.mlir 2>&1
```

Check for SSA dominance errors:
```bash
grep -i "error\|dominance" /tmp/volume_integral_dbs.mlir
```

Check db_alloc/db_free placement:
```bash
grep -n "db_alloc\|db_free" /tmp/volume_integral_dbs.mlir
```

### Stage 3: concurrency
```bash
carts run /tmp/volume_integral.mlir --concurrency > /tmp/volume_integral_conc.mlir 2>&1
```

## 5. Build and run the example

```bash
make clean && make all
./volume_integral_arts
```

---

## Bug Analysis: SSA Dominance Violation

### The Problem

The volume-integral example was failing at CARTS pipeline stage 4 (ARTS transformation) with:
```
error: operand #0 does not dominate this use
```

### Root Cause

The `float buffer[N_QUAD]` is a valid fixed-size stack array allocated INSIDE the `arts.for` parallel loop:

```c
#pragma omp parallel for schedule(static)
for (int elem = 0; elem < N_ELEMENTS; ++elem) {
    float buffer[N_QUAD];  // Thread-local stack allocation
    ...
}
```

**After `--openmp-to-arts`** the buffer is correctly placed:
```mlir
"arts.for"(%c0, %c64, %c1) ({
^bb0(%arg0: index):
  %alloca = memref.alloca() : memref<36xf32>  // Buffer INSIDE arts.for
  ...
}) : (index, index, index) -> ()
```

**After `--create-dbs`** (BUGGY): `db_alloc` was inside the loop, but `db_free` was placed OUTSIDE:
```mlir
"arts.edt"(...) ({
^bb0(...):
  "arts.for"(%15, %14, %13) ({
  ^bb0(%arg4: index):
    %64:2 = "arts.db_alloc"(...) // <-- INSIDE arts.for
    ...
  }) : (index, index, index) -> ()
  "arts.db_free"(%64#0)  // <-- OUTSIDE arts.for - %64 NOT VISIBLE HERE!
  "arts.db_free"(%64#1)  // <-- SSA DOMINANCE VIOLATION!
  ...
})
```

### The Bug Location

File: `/Users/randreshg/Documents/carts/lib/arts/Passes/CreateDbs.cpp`
Function: `insertDbFreeForDbAlloc()` (around line 946)

**Original code (buggy):**
```cpp
/// Determine where to insert DbFreeOp based on where dbAlloc is located
Operation *insertionPoint = nullptr;

/// Check if dbAlloc is within an EDT
if (EdtOp parentEdt = dbAlloc->getParentOfType<EdtOp>()) {
    /// Insert before EDT terminator
    Block &edtBlock = parentEdt.getBody().front();
    insertionPoint = edtBlock.getTerminator();  // <-- WRONG!
} else {
    Block *allocBlock = dbAlloc->getBlock();
    insertionPoint = allocBlock->getTerminator();
}
```

This code found the parent EDT and inserted `db_free` at the EDT terminator, but the `db_alloc` was created inside a nested `arts.for` loop. The `%64` SSA value is only visible inside the `arts.for` block - not at the EDT level!

### The Fix

```cpp
/// Determine where to insert DbFreeOp based on where dbAlloc is located
/// Insert at the end of the block containing dbAlloc to ensure SSA dominance
/// (the SSA value from db_alloc is only visible within its containing block)
Block *allocBlock = dbAlloc->getBlock();
Operation *insertionPoint = allocBlock->getTerminator();
```

The fix simply uses `dbAlloc->getBlock()->getTerminator()` - placing `db_free` in the same block as `db_alloc`, which preserves SSA dominance rules.

---

## Verification Commands

After rebuilding CARTS with the fix:

```bash
# 1. Rebuild CARTS
cd ~/Documents/carts && carts build

# 2. Test volume-integral
cd ~/Documents/carts/external/carts-benchmarks/seissol/volume-integral
make clean && make all

# 3. Verify no dominance errors in create-dbs stage
carts cgeist volume_integral.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I../common > /tmp/vi.mlir
carts run /tmp/vi.mlir --create-dbs 2>&1 | grep -i "error\|dominance"

# 4. Check db_free is now inside arts.for
carts run /tmp/vi.mlir --create-dbs 2>&1 | grep -B5 -A5 "db_free"
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

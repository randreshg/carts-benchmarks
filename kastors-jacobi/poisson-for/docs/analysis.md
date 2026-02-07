# Poisson-For Stencil Example Analysis

This document walks through the CARTS pipeline for the poisson-for benchmark to help diagnose correctness issues.

## Current Status

**Working**: ARTS and OpenMP checksums match (small/medium/large).

### Current Partitioning Summary (No ESD)

The current pipeline **does not use ESD/byte-slice dependencies** here.
DbPartitioning uses **element-wise row DBs** for the stencil source (`u`) and
**chunked DBs** for uniform read/write arrays:

- `u`: **element_wise** (one DB per row) for stencil reads
- `unew`: **chunked** (one DB per chunk) for uniform writes
- `f`: **chunked** (one DB per chunk) for uniform reads

Halo exchange is handled by **fetching neighbor row DBs**, not by byte-offset
slices.

---

## 1. Navigate to the poisson-for directory

```bash
cd /opt/carts/external/carts-benchmarks/kastors-jacobi/poisson-for
```

## 2. Build CARTS if any changes were made

```bash
carts build
```

## 3. Generate MLIR files

If there is no poisson-for.mlir, run:

```bash
# Sequential version for metadata collection
carts cgeist poisson-for.c -O0 --print-debug-info -S --raise-scf-to-affine -I../include -DSIZE=256 &> poisson-for_seq.mlir

# Collect metadata
carts run poisson-for_seq.mlir --collect-metadata &> poisson-for_arts_metadata.mlir

# Parallel version with OpenMP annotations
carts cgeist poisson-for.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I../include -DSIZE=256 &> poisson-for.mlir
```

---

## 4. Pipeline Stage Analysis

### Stage 1: OpenMP to ARTS conversion

```bash
carts run poisson-for.mlir --stop-after=openmp-to-arts 2>&1 | head -200
```

Look for:
- `arts.parallel_for` operations replacing `omp.parallel_for`
- Three arrays: `u`, `unew`, `f` (rhs)

### Stage 2: Create DBs

```bash
carts run poisson-for.mlir --stop-after=create-dbs --debug-only=create_dbs 2>&1 | head -200
```

Expected: Three `arts.db_alloc` operations for the stencil arrays.

### Stage 3: DB Partitioning (CRITICAL)

```bash
carts run poisson-for.mlir --stop-after=concurrency-opt --debug-only=db_partitioning 2>&1 | head -300
```

**Key outputs to verify**:

1. **Partition mode detection**:
```
Access patterns: hasUniform=1, hasStencil=1, hasIndexed=0, isMixed=1
Decision: mode=Stencil, outerRank=1, innerRank=1
Stencil mode (early): haloLeft=1, haloRight=1
```

2. **Chunk sizing**:
   - With nodeCount=3 and SIZE=256:
   - Expected chunks: ~3 chunks
   - Each chunk: ~85 rows + halo regions

3. **EDT structure**:
   - Should see 3 types of EDTs:
   - Copy EDT (1 dep)
   - Interior sweep EDT (4 deps: owned + left_halo + right_halo + rhs)
   - Boundary sweep EDT (5 deps: owned + left_halo + right_halo + prev_result + rhs)

### Stage 4: Pre-lowering (EDT Functions)

```bash
carts run poisson-for.mlir --stop-after=pre-lowering 2>&1 > /tmp/pre-lowering.mlir
grep -n "func.func.*__arts_edt" /tmp/pre-lowering.mlir
```

Examine each EDT function for:
- Number of dependencies accessed
- Buffer selection logic (3-buffer model)
- Index calculations

### Stage 5: ARTS to LLVM conversion

```bash
carts run poisson-for.mlir --stop-after=arts-to-llvm --debug-only=convert_arts_to_llvm 2>&1 | head -200
```

Verify:
- RecordDep patterns correctly counting slots
- Each EDT has correct number of datablocks

---

## 5. Stencil Pattern Analysis

The Jacobi stencil for poisson-for uses:
```c
unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - h*h*f[i][j])
```

### Partitioning Model

With stencil partitioning:
```
Chunk 0: rows [0, chunkSize)
  - No left halo (boundary)
  - Right halo: row [chunkSize] from chunk 1

Chunk 1: rows [chunkSize, 2*chunkSize)
  - Left halo: row [chunkSize-1] from chunk 0
  - Right halo: row [2*chunkSize] from chunk 2

Chunk 2: rows [2*chunkSize, N)
  - Left halo: row [2*chunkSize-1] from chunk 1
  - No right halo (boundary)
```

### Halo Access Model (Row DBs, Not ESD)

Each stencil EDT acquires row DBs for:
- **Owned rows**: rows in the worker's chunk
- **Left halo row**: `row = chunkStart - 1` (when in bounds)
- **Right halo row**: `row = chunkEnd` (when in bounds)

Because `u` is element-wise, each row is its own DB. The stencil reads are
resolved by selecting the appropriate **row DB** directly, without byte slices.

---

## 6. Execute and Verify

### Build both versions

```bash
make SIZE=small clean
make SIZE=small          # Builds ARTS version
make SIZE=small openmp   # Builds OMP version
```

### Run ARTS version

```bash
# Make sure arts.cfg matches runtime config!
cat arts.cfg | grep nodeCount  # Should match runtime nodes

ARTS_CONFIG=/opt/carts/external/carts-benchmarks/.generated_configs/arts_4t_3n.cfg ./poisson-for_arts
```

### Run OMP version

```bash
OMP_NUM_THREADS=4 ./build/poisson-for_omp
```

### Expected matching checksums

Both should produce identical checksum values.

---

## 7. Debugging Checklist

### Config Mismatch

**CRITICAL**: The `arts.cfg` used at compile time MUST match the runtime config!

```bash
# Check compile-time config
grep nodeCount arts.cfg

# Check runtime config
grep nodeCount /opt/carts/external/carts-benchmarks/.generated_configs/arts_4t_3n.cfg
```

If these don't match, recompile:
```bash
# Edit arts.cfg to match runtime
sed -i 's/nodeCount=.*/nodeCount=3/' arts.cfg
make SIZE=small clean && make SIZE=small
```

### Slot Recording vs Access

Add debug prints to trace slot indices:

**Recording side** (ConvertArtsToLLVM.cpp):
```cpp
ARTS_DEBUG("RecordDep: slot=" << slot << " dbGuid=" << dbGuid);
```

**Access side** (EdtLowering.cpp):
```cpp
ARTS_DEBUG("Access: depIndex=" << depIndex << " baseOffset=" << baseOffset);
```

Run with debug:
```bash
carts run poisson-for.mlir --debug-only=convert_arts_to_llvm,edt_lowering 2>&1 | grep -E "RecordDep|Access"
```

### Stencil Buffer Selection

Check DbStencilRewriter.cpp for 3-buffer selection logic:
```bash
carts run poisson-for.mlir --stop-after=pre-lowering --debug-only=db_stencil 2>&1 | head -200
```

Verify:
- `localRow` calculation is correct
- Buffer boundaries (haloLeft, chunkSize) are correct
- arith.select operations choose correct buffer

---

## 8. Known Issues

### Issue 1: Compile/Runtime Config Mismatch

**Symptom**: Wrong checksum even with correct algorithm
**Cause**: arts.cfg nodeCount at compile time differs from runtime
**Fix**: Ensure arts.cfg matches runtime config before compilation

### Issue 2: ESD Byte Offset/Size (Not Active)

**Note**: ESD byte offsets are not currently generated for this example because
DbPartitioning does not emit halo metadata. All halo access uses row DBs.

---

## 9. Quick Verification Commands

```bash
# Full pipeline with all debug output
carts run poisson-for.mlir --all-stages -o ./stages/ 2>&1 | tee logs/full_pipeline.log

# Compare specific stage outputs
diff stages/10_concurrency.mlir stages/11_concurrency-opt.mlir | head -100

# Run benchmark suite
carts benchmarks run kastors-jacobi/poisson-for --nodes 3 --verbose

# Check benchmark results
cat /opt/carts/external/carts-benchmarks/results/benchmark_results_*.json | jq '.results[].verification'
```

---

## Appendix: Source Code Structure

**poisson-for.c** key functions:
- `rhs()`: Initialize the right-hand side array `f`
- `sweep()`: Jacobi iteration kernel (stencil computation)
- `main()`: Orchestrates iterations and computes checksum

**Stencil kernel** (simplified):
```c
#pragma omp parallel for
for (int i = 1; i < n-1; i++) {
    for (int j = 1; j < n-1; j++) {
        unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] +
                             u[i][j-1] + u[i][j+1] - h*h*f[i][j]);
    }
}
```

The boundary conditions (i=0, i=n-1, j=0, j=n-1) are handled separately, copying from `u` to `unew` unchanged.

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

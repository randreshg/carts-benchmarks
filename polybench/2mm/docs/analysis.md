# 2mm example analysis (Block Partitioning)

Walk through these steps to understand how CARTS transforms matrix multiplication kernels using block partitioning.

## Key Differences: Block vs Stencil Partitioning

| Aspect | jacobi-for (Stencil) | 2mm (Block) |
|--------|----------------------|-------------|
| Access pattern | Neighbor dependencies | Row-wise independent |
| Partition mode | `stencil` with halos | `block` for partitionable, `coarse` for non-partitionable |
| Halo handling | `left_halo_arg_idx` / `right_halo_arg_idx` | Not needed |
| Key insight | Detects i-1, i+1 accesses | Detects k-dimension reduction |

## 1. Getting Started

1. **Navigate to the 2mm example directory:**

   ```bash
   cd /opt/carts/external/carts-benchmarks/polybench/2mm
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no 2mm.mlir, run:

   ```bash
   carts cgeist 2mm.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> 2mm_seq.mlir
   carts run 2mm_seq.mlir --collect-metadata &> 2mm_arts_metadata.mlir
   carts cgeist 2mm.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> 2mm.mlir
   ```

## 2. Source Code Analysis

The 2mm benchmark performs `D = alpha*A*B*C + beta*D` (two matrix multiplications):

```c
/* kernel_2mm - D := alpha*A*B*C + beta*D */
static void kernel_2mm(int ni, int nj, int nk, int nl,
                       DATA_TYPE alpha, DATA_TYPE beta,
                       DATA_TYPE POLYBENCH_2D(tmp, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(C, NJ, NL, nj, nl),
                       DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
  int i, j, k;

  /* First matrix multiplication: tmp = alpha * A * B */
  #pragma omp parallel for private(j, k)
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
      tmp[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_NK; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];  // k-dimension reduction
    }
  }

  /* Second matrix multiplication: D = beta*D + tmp * C */
  #pragma omp parallel for private(j, k)
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NL; j++) {
      D[i][j] *= beta;
      for (k = 0; k < _PB_NJ; ++k)
        D[i][j] += tmp[i][k] * C[k][j];  // k-dimension reduction
    }
  }
}
```

**Access patterns:**
- `tmp[i][j]`, `A[i][k]`, `D[i][j]` - Row-wise access (partitionable on i)
- `B[k][j]`, `C[k][j]` - k-dimension access (full matrix needed for reduction)

**Key insight:** Arrays B and C are accessed with the loop variable `k` in the first dimension, meaning each task needs the entire matrix for the k-dimension summation. These arrays cannot benefit from block partitioning.

## 3. Pipeline Stage 1: CreateDbs

**Command:**
```bash
carts run 2mm.mlir --create-dbs &> 2mm_create-dbs.mlir
```

At this stage, all arrays start with `<coarse>` partition mode:

```mlir
/// All arrays allocated as COARSE - single entry covering entire array
/// No explicit partition hints from user

/// tmp array - intermediate result
%guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 37 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// A array - first input matrix
%guid_3, %ptr_4 = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 41 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// B array - second input matrix (will stay coarse)
%guid_5, %ptr_6 = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 42 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// C array - third input matrix (will stay coarse)
%guid_7, %ptr_8 = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 43 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// D array - output matrix
%guid_9, %ptr_10 = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 44 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

## 4. Pipeline Stage 2: DbPartitioning (concurrency-opt)

**Command:**
```bash
carts run 2mm.mlir --concurrency-opt &> 2mm_concurrency-opt.mlir
```

DbPartitioning analyzes memory access patterns and makes partitioning decisions:

### Arrays with Block Partitioning (tmp, A, D)

```mlir
/// tmp array -> <block> mode (row-wise write access)
%guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%c0_i32 : i32)
    sizes[%13] elementType(f64) elementSizes[%c8, %c128]
    {arts.id = 37 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// A array -> <block> mode (row-wise read access)
%guid_3, %ptr_4 = arts.db_alloc[<in>, <heap>, <read>, <block>] route(%c0_i32 : i32)
    sizes[%13] elementType(f64) elementSizes[%c8, %c128]
    {arts.id = 41 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// D array -> <block> mode (row-wise read-modify-write)
%guid_9, %ptr_10 = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%c0_i32 : i32)
    sizes[%13] elementType(f64) elementSizes[%c8, %c128]
    {arts.id = 44 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

### Arrays with Coarse Partitioning (B, C)

```mlir
/// B array -> <coarse> mode (k-dimension access, non-partitionable)
%guid_5, %ptr_6 = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 42 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// C array -> <coarse> mode (k-dimension access, non-partitionable)
%guid_7, %ptr_8 = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c128]
    {arts.id = 43 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

### Block Acquires with Partitioning

```mlir
/// Block acquire for tmp (write)
%guid_12, %ptr_13 = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?x?xf64>>)
    partitioning(<block>, offsets[%38], sizes[%c8]),
    offsets[%44], sizes[%51]
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// Block acquire for A (read)
%guid_14, %ptr_15 = arts.db_acquire[<in>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<?x?xf64>>)
    partitioning(<block>, offsets[%38], sizes[%c8]),
    offsets[%44], sizes[%51]
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// Coarse acquire for B (full matrix)
%guid_16, %ptr_17 = arts.db_acquire[<in>] (%guid_5 : memref<?xi64>, %ptr_6 : memref<?xmemref<?x?xf64>>)
    offsets[%c0], sizes[%c1]
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

## 5. Block Partitioning Explained

### Partitioning Decision Summary

| Array | ID | Partition Mode | Block Size | Access Mode | Reason |
|-------|-----|----------------|------------|-------------|--------|
| tmp | 37 | `<block>` | 8x128 | `<inout>` | Row-wise write access |
| A | 41 | `<block>` | 8x128 | `<in>` | Row-wise read, partitionable |
| B | 42 | `<coarse>` | N/A | `<in>` | Input mode + non-partitionable access |
| C | 43 | `<coarse>` | N/A | `<in>` | Input mode + non-partitionable access |
| D | 44 | `<block>` | 8x128 | `<inout>` | Row-wise read-modify-write |

### Why Row-Wise Arrays Get Block Partitioning

Arrays `tmp`, `A`, and `D` are accessed with pattern `array[i][...]` where `i` is the parallel loop variable:
- Each iteration only accesses rows belonging to its chunk
- No data dependencies between iterations on the i-dimension
- Can be safely partitioned into blocks of 8 rows each

### Why k-Dimension Arrays Stay Coarse

Arrays `B` and `C` are accessed with pattern `array[k][j]` where `k` is the inner reduction loop variable:
- Each task needs to read ALL rows of B and C for the k-dimension summation
- Block partitioning would require each task to acquire every block
- Keeping them coarse (single entry) is more efficient

### Block Size Determination

The block size of 8 rows is determined by the heuristics based on:
- Loop bounds and trip count
- Number of available parallel units
- Memory access patterns

### How `partitioning(<block>, offsets[...], sizes[...])` Works

```mlir
partitioning(<block>, offsets[%38], sizes[%c8])
```

- `<block>` - Partition mode indicating row-wise chunking
- `offsets[%38]` - Block index computed from loop iteration (iter / blockSize)
- `sizes[%c8]` - Each block contains 8 rows

## 6. Debug Output Analysis

**Command:**
```bash
carts run 2mm.mlir --concurrency-opt --debug-only=db_partitioning 2>&1 | grep -E "(Decision|mode|canBlock|Partition|coarse|re-analyzing|Plan)"
```

### For Partitionable Arrays (tmp, A, D)

```
Partition mode is coarse - re-analyzing with loop structure
Acquire[0]: mode=2, canEW=0, canBlock=1, acquireAttr=1
Decision: mode=Block, outerRank=1, innerRank=1
Plan created: blockSizes=1, numPartitionedDims=1
Set partition attribute: Block
```

Key observations:
- `canBlock=1` - Access pattern supports block partitioning
- `mode=Block` - Final decision is block mode
- `outerRank=1, innerRank=1` - 1D partitioning on 2D array

### For Non-Partitionable Input Arrays (B, C)

```
Partition mode is coarse - re-analyzing with loop structure
Acquire[0]: mode=0, canEW=0, canBlock=0, acquireAttr=1
Per-acquire voting: canEW=0, canBlock=0, modesConsistent=1
Coarse mode - no partitioning needed
```

Key observations:
- `canBlock=0` - Access pattern does NOT support block partitioning
- The k-dimension access `B[k][j]` is detected as non-partitionable
- Arrays stay `<coarse>` mode with single entry

## 7. Comparison with Other Benchmarks

### vs jacobi-for (Stencil Mode)

| Aspect | 2mm (Block) | jacobi-for (Stencil) |
|--------|-------------|----------------------|
| Pattern | Matrix multiplication | 5-point stencil |
| Neighbor access | None | u[i-1], u[i+1] |
| Halo handling | Not needed | left/right halo args |
| Dependencies | Independent rows | Neighbor dependencies |

### vs gemm, gemver (Similar Block Pattern)

The 2mm benchmark is similar to other BLAS-style operations:
- `gemm` - Single matrix multiplication, same block partitioning
- `gemver` - Vector operations with similar row-wise access
- All share the pattern of some arrays being partitionable (row access) and some staying coarse (reduction dimension)

## 8. Verification

**Run the benchmark:**
```bash
carts benchmarks run polybench/2mm
```

**Expected result:**
- Correctness: YES (checksum matches)
- Speedup: ~1.8x over OMP

**Quick execution check:**
```bash
carts execute 2mm.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
./2mm_arts
```

## Troubleshooting

### If Correctness Fails

1. **Check partition modes:**
   ```bash
   carts run 2mm.mlir --concurrency-opt | grep -E "db_alloc.*block|db_alloc.*coarse"
   ```
   Should show `<block>` for tmp, A, D and `<coarse>` for B, C.

2. **Verify block sizes:**
   ```bash
   carts run 2mm.mlir --concurrency-opt | grep "elementSizes"
   ```
   Block arrays should have reduced first dimension (e.g., 8 instead of full size).

3. **Debug partitioning decisions:**
   ```bash
   carts run 2mm.mlir --concurrency-opt --debug-only=db_partitioning 2>&1 | grep -E "canBlock|Decision"
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

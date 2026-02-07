# jacobi-for example analysis (ESD / Stencil Pipeline)

Walk through these steps to understand how the ESD (Extended Stencil Dependency) infrastructure transforms implicit stencil patterns into explicit halo-based parallelism.

## Key Differences: jacobi-for vs poisson-task

| Aspect | poisson-task | jacobi-for |
|--------|--------------|------------|
| OpenMP construct | `#pragma omp task depend(...)` | `#pragma omp parallel for` |
| Dependencies | **Explicit** in depend clause | **Implicit** in stencil pattern |
| Partition mode | `fine_grained` | `stencil` (ESD) + `block` |
| Halo handling | Multi-entry expansion | `left_halo_arg_idx` / `right_halo_arg_idx` |

## 1. Getting Started

1. **Navigate to the jacobi-for example directory:**

   ```bash
   cd /opt/carts/external/carts-benchmarks/kastors-jacobi/jacobi-for
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no jacobi-for.mlir, run:

   ```bash
   carts cgeist jacobi-for.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> jacobi-for_seq.mlir
   carts run jacobi-for_seq.mlir --collect-metadata &> jacobi-for_arts_metadata.mlir
   carts cgeist jacobi-for.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> jacobi-for.mlir
   ```

## 2. Source Code Analysis

The jacobi-for benchmark uses `#pragma omp parallel for` with **implicit** stencil dependencies:

```c
// Save the current estimate
#pragma omp parallel for private(j)
for (i = 0; i < nx; i++) {
  for (j = 0; j < ny; j++) {
    u[i][j] = unew[i][j];
  }
}

// Compute a new estimate
#pragma omp parallel for private(j)
for (i = 0; i < nx; i++) {
  for (j = 0; j < ny; j++) {
    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
      unew[i][j] = f[i][j];
    } else {
      // IMPLICIT STENCIL: accesses u[i-1][j], u[i][j+1], u[i][j-1], u[i+1][j]
      unew[i][j] = 0.25 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] +
                           u[i + 1][j] + f[i][j] * dx * dy);
    }
  }
}
```

**Key insight:** Unlike poisson-task (which has explicit `depend(in: u[i-1], u[i], u[i+1])`), jacobi-for has **no dependency annotations**. The compiler must detect the stencil pattern from memory access analysis.

## 3. Pipeline Stage 1: CreateDbs

**Command:**
```bash
carts run jacobi-for.mlir --create-dbs &> jacobi-for_create-dbs.mlir
```

At this stage, all arrays start with `<coarse>` partition mode since there are no explicit partition hints:

```mlir
/// All arrays allocated as COARSE - single-entry covering entire array
/// No explicit partition hints from user (unlike poisson-task's depend clause)

/// f array - input source term
%guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c256]
    {arts.id = 37 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// u array - previous iteration values (will become stencil mode)
%guid_3, %ptr_4 = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c256]
    {arts.id = 40 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// unew array - current iteration output
%guid_5, %ptr_6 = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%c0_i32 : i32)
    sizes[%c1] elementType(f64) elementSizes[%9, %c256]
    {arts.id = 41 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

The acquires also start as coarse, acquiring the entire array:

```mlir
/// Inside the parallel for EDT - coarse acquire covering whole array
arts.edt <parallel> <internode> route(%c0_i32) (%ptr_8, %ptr_10) ... {
  arts.for(%c0) to(%c256) step(%c1) {{
  ^bb0(%arg3: index):
    /// Access pattern: unew[arg3][arg4] = u[arg3][arg4]
    %31 = arts.db_ref %arg1[%c0] : memref<?xmemref<?x?xf64>> -> memref<?x?xf64>
    %32 = memref.load %31[%arg3, %arg4] : memref<?x?xf64>
    ...
  }}
}
```

## 4. Pipeline Stage 2: DbPartitioning (concurrency-opt)

**Command:**
```bash
carts run jacobi-for.mlir --concurrency-opt &> jacobi-for_concurrency-opt.mlir
```

DbPartitioning analyzes memory access patterns and detects the implicit stencil:

```mlir
/// f array -> <block> mode (no stencil pattern, regular block access)
%guid, %ptr = arts.db_alloc[<in>, <heap>, <read>, <block>] route(%c0_i32 : i32)
    sizes[%13] elementType(f64) elementSizes[%c16, %c256]
    {arts.id = 37 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// u array -> <stencil> mode (ESD detected: accesses u[i-1], u[i], u[i+1])
%guid_3, %ptr_4 = arts.db_alloc[<inout>, <heap>, <write>, <stencil>] route(%c0_i32 : i32)
    sizes[%13] elementType(f64) elementSizes[%c16, %c256]
    {arts.id = 40 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// unew array -> <block> mode (simple output, no neighbor access)
%guid_5, %ptr_6 = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%c0_i32 : i32)
    sizes[%13] elementType(f64) elementSizes[%c16, %c256]
    {arts.id = 41 : i64} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

### Stencil Acquires with Halo Infrastructure

For the stencil computation EDT, the `u` array acquires show the ESD infrastructure:

```mlir
/// Left halo acquire: u[block-1] - acquires last row of previous block
%guid_9, %ptr_10 = arts.db_acquire[<in>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<?x?xf64>>)
    partitioning(<stencil>),
    offsets[%51], sizes[%c1]           /// Block offset: (block_idx - 1)
    bounds_valid(%52)                   /// Boundary check: block_idx != 0
    element_offsets[%c15, %c0]          /// Within block: last row (index 15)
    element_sizes[%c1, %c256]           /// Single row, full width
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// Right halo acquire: u[block+1] - acquires first row of next block
%guid_11, %ptr_12 = arts.db_acquire[<in>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<?x?xf64>>)
    partitioning(<stencil>),
    offsets[%53], sizes[%c1]           /// Block offset: (block_idx + 1)
    bounds_valid(%54)                   /// Boundary check: block_idx != (numBlocks - 1)
    element_offsets[%c0, %c0]           /// Within block: first row (index 0)
    element_sizes[%c1, %c256]           /// Single row, full width
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

/// Main block acquire with halo references
%guid_15, %ptr_16 = arts.db_acquire[<in>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<?x?xf64>>)
    partitioning(<stencil>),
    offsets[%43], sizes[%c1]           /// Main block offset
    {arts.id = 124 : i64, left_halo_arg_idx = 3 : index,    /// EDT arg index for left halo
     right_halo_arg_idx = 4 : index}   /// EDT arg index for right halo
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

### Block-Mode Acquires (f, unew)

```mlir
/// f and unew use simple block partitioning (no neighbor dependencies)
%guid_7, %ptr_8 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?x?xf64>>)
    partitioning(<block>, offsets[%37], sizes[%c16]),
    offsets[%43], sizes[%50]
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

%guid_9, %ptr_10 = arts.db_acquire[<out>] (%guid_5 : memref<?xi64>, %ptr_6 : memref<?xmemref<?x?xf64>>)
    partitioning(<block>, offsets[%37], sizes[%c16]),
    offsets[%43], sizes[%50]
    {arts.id = 124 : i64}
    -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
```

## 5. ESD Infrastructure Explained

### How DbPartitioning Detects Implicit Stencil

1. **Initial coarse mode:** All arrays start as `<coarse>` with no partition hints

2. **Re-analysis with loop structure:** When partition mode is coarse, DbPartitioning re-analyzes using loop context:
   - Examines memory access patterns within `arts.for` loops
   - Looks for neighbor accesses (e.g., `%arg3 + 1`, `%arg3 - 1`)

3. **Inconsistent mode detection:** When the same array has different access patterns:
   - Some accesses are at current index (`u[i][j]`)
   - Some accesses are at neighbor indices (`u[i-1][j]`, `u[i+1][j]`)
   - This triggers "inconsistent partition modes" - can't be pure block mode

4. **Stencil mode decision:** When inconsistency is detected and accesses form a stencil pattern:
   - `mode=Stencil` is chosen
   - `outerRank=1, innerRank=1` for 2D array with 1D partitioning
   - `haloLeft=1, haloRight=1` for 5-point stencil (accesses i-1 and i+1)

### Key ESD Attributes

| Attribute | Purpose |
|-----------|---------|
| `partitioning(<stencil>)` | Marks this acquire as part of stencil pattern |
| `bounds_valid(%cond)` | Condition for boundary checking (skip halo at edges) |
| `element_offsets[row, col]` | Offset within block for halo data |
| `element_sizes[rows, cols]` | Size of halo region (typically 1 row) |
| `left_halo_arg_idx` | EDT argument index containing left neighbor data |
| `right_halo_arg_idx` | EDT argument index containing right neighbor data |

## 6. Debug Output Analysis

**Command:**
```bash
carts run jacobi-for.mlir --concurrency-opt --debug-only=db_partitioning 2>&1 | grep -E "(Decision|mode|halo|Partition|stencil|inconsistent|coarse|re-analyzing)"
```

**Key debug lines explaining ESD detection:**

```
[DEBUG] [db_partitioning]   Partition mode is coarse - re-analyzing with loop structure
[DEBUG] [db_partitioning]   Inconsistent partition modes across acquires
[DEBUG] [db_partitioning]     Acquire[0]: mode=0, canEW=0, canBlock=1, acquireAttr=1
[DEBUG] [db_partitioning]     Acquire[1]: mode=1, canEW=0, canBlock=1, acquireAttr=1
[DEBUG] [db_partitioning]   Per-acquire voting: canEW=0, canBlock=1, modesConsistent=0
[DEBUG] [db_partitioning]   Partition dim counts: min=1, max=1
[DEBUG] [db_partitioning]   Block/stencil mode with indexed/uniform patterns - extracting N-D block sizes
[DEBUG] [db_partitioning]   Decision: mode=Stencil, outerRank=1, innerRank=1
[DEBUG] [db_partitioning]   Stencil mode (early): haloLeft=1, haloRight=1
[DEBUG] [db_partitioning]   Plan created: blockSizes=1, numPartitionedDims=1
```

### Interpreting the Debug Output

1. **"Partition mode is coarse - re-analyzing with loop structure"**
   - Initial pass found coarse mode, triggering deeper analysis

2. **"Inconsistent partition modes across acquires"**
   - The `u` array has multiple acquires with different access patterns
   - `mode=0` (block-like) and `mode=1` (stencil-like) detected

3. **"modesConsistent=0"**
   - Confirms the inconsistency that triggers stencil handling

4. **"Decision: mode=Stencil, outerRank=1, innerRank=1"**
   - Final decision: use stencil mode
   - 1D partitioning (outerRank=1) on 2D array (remaining innerRank=1)

5. **"Stencil mode (early): haloLeft=1, haloRight=1"**
   - Detected 1-element halo on both sides
   - Matches the `u[i-1]` and `u[i+1]` accesses

## 7. ESD vs Explicit Dependencies Comparison

### poisson-task (Explicit Dependencies)

```c
// User explicitly declares neighbor dependencies
#pragma omp task depend(in: f[i], u[i-1], u[i], u[i+1]) depend(out: unew[i])
```

Result: **Multi-entry expansion** - creates 3 separate acquires for u[i-1], u[i], u[i+1]

### jacobi-for (Implicit Dependencies via ESD)

```c
// No depend clause - compiler must detect stencil from memory accesses
#pragma omp parallel for
for (i = 0; i < nx; i++) {
  ... u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] ...
}
```

Result: **Stencil mode** - creates halo acquires with `left_halo_arg_idx` / `right_halo_arg_idx` linking

### Key Architectural Difference

| Approach | How Dependencies Are Handled |
|----------|------------------------------|
| Multi-entry expansion (poisson-task) | Each neighbor is a separate EDT argument |
| ESD stencil mode (jacobi-for) | Main block + halo arguments with index linking |

## 8. Verification

**Run the benchmark:**
```bash
carts benchmarks run kastors-jacobi/jacobi-for --trace
```

**Expected result:**
- Correctness: YES (checksum matches)
- Kernel and E2E timings reported

**Quick execution check:**
```bash
carts execute jacobi-for.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
./jacobi-for_arts
```

## Troubleshooting

### If Correctness Fails

1. **Check stencil detection:**
   ```bash
   carts run jacobi-for.mlir --concurrency-opt | grep "partition_mode"
   ```
   Should show `<stencil>` for the `u` array.

2. **Verify halo attributes:**
   ```bash
   carts run jacobi-for.mlir --concurrency-opt | grep -E "left_halo|right_halo|element_offsets"
   ```
   Should show halo index attributes on stencil acquires.

3. **Check boundary handling:**
   ```bash
   carts run jacobi-for.mlir --concurrency-opt | grep "bounds_valid"
   ```
   Halo acquires should have bounds checks to skip at array boundaries.

4. **Debug halo generation:**
   ```bash
   carts run jacobi-for.mlir --concurrency-opt --debug-only=db_partitioning 2>&1 | grep -i halo
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

# poisson-task example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the poisson-task example directory:**

   ```bash
   cd /opt/carts/external/carts-benchmarks/kastors-jacobi/poisson-task
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist poisson-task.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> poisson-task_seq.mlir
      carts run poisson-task_seq.mlir --collect-metadata &> poisson-task_arts_metadata.mlir
      carts cgeist poisson-task.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> poisson-task.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the create-dbs pipeline
    ```bash
      carts run poisson-task.mlir --create-dbs &> poisson-task_create-dbs.mlir
    ```

    Analyze the comments within the summarized output.
    ```mlir
   module attributes {...} {
   ...
   func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      ...
      /// Check that all the DbAllocOps respected the control deps in the original code. In the .c code we have
      /// row dependencies
      %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>] route(%c0_i32 : i32) sizes[%13] elementType(f64) elementSizes[%c100] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
      %guid_4, %ptr_5 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%13] elementType(f64) elementSizes[%c100] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
      %guid_6, %ptr_7 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%13] elementType(f64) elementSizes[%c100] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
      %alloc = memref.alloc(%13) : memref<?x100xf64>
      scf.for %arg0 = %c0 to %c100 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
         /// Notice that we dbref the outer dimension of the db and then store the value to the inner dimension
         %31 = arts.db_ref %ptr[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %31[%arg1] : memref<?xf64>
         %32 = arts.db_ref %ptr_5[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %32[%arg1] : memref<?xf64>
         %33 = arts.db_ref %ptr_7[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %33[%arg1] : memref<?xf64>
         memref.store %cst_3, %alloc[%arg0, %arg1] : memref<?x100xf64>
         } {...}
      } {...}
      ...
      %17 = arts.epoch {
         scf.for %arg0 = %c1 to %c11 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
            /// Acquire unew[i] - we acquired the whole row unew[i] and added the index %arg1 to the indices
            %guid_8, %ptr_9 = arts.db_acquire[<in>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// Acquire u[i] - we acquired the whole row u[i] and added the index %arg1 to the indices
            %guid_10, %ptr_11 = arts.db_acquire[<out>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// EDT that computes u[i] = unew[i]
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_9, %ptr_11) : memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>> {
            ^bb0(%arg2: memref<?xmemref<memref<?xf64>>>, %arg3: memref<?xmemref<memref<?xf64>>>):
               scf.for %arg4 = %c0 to %c100 step %c1 {
               /// Load unew[i][j]
               %31 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
               %32 = memref.load %31[%arg4] : memref<?xf64>
               /// Load u[i][j]
               %33 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
               /// Store u[i][j] = unew[i][j]
               memref.store %32, %33[%arg4] : memref<?xf64>
               } {...}
               arts.db_release(%arg2) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg3) : memref<?xmemref<memref<?xf64>>>
            }
         } {...}
         scf.for %arg1 = %c0 to %c100 step %c1 {
            ...
            /// IN: Acquire f[i] - we acquired the whole row f[i] and added the index %arg1 to the indices
            %guid_8, %ptr_9 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// OUT: Acquire unew[i] - we acquired the whole row unew[i] and added the index %arg1 to the indices
            %guid_10, %ptr_11 = arts.db_acquire[<out>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// IN: Acquire u[i-1] - we acquired the whole row u[i-1] and added the index %32 to the indices
            %guid_12, %ptr_13 = arts.db_acquire[<in>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%32] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// IN: Acquire u[i] - we acquired the whole row u[i] and added the index %arg1 to the indices
            %guid_14, %ptr_15 = arts.db_acquire[<in>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// IN: Acquire u[i+1] - we acquired the whole row u[i+1] and added the index %33 to the indices
            %guid_16, %ptr_17 = arts.db_acquire[<in>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%33] offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// EDT that computes u[i][j] = 0.25 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy)
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_9, %ptr_11, %ptr_13, %ptr_15, %ptr_17) : memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>> {
            ^bb0(%arg2: memref<?xmemref<memref<?xf64>>>, %arg3: memref<?xmemref<memref<?xf64>>>, %arg4: memref<?xmemref<memref<?xf64>>>, %arg5: memref<?xmemref<memref<?xf64>>>, %arg6: memref<?xmemref<memref<?xf64>>>):
               scf.for %arg7 = %c0 to %c100 step %c1 {
               %35 = arith.index_cast %arg7 : index to i32
               %36 = arith.cmpi eq, %arg1, %c0 : index
               %37 = scf.if %36 -> (i1) {
                  scf.yield %true : i1
               } else {
                  %42 = arith.cmpi eq, %35, %c0_i32 : i32
                  scf.yield %42 : i1
               }
               %38 = arith.xori %37, %true : i1
               %39 = arith.andi %38, %34 : i1
               %40 = arith.ori %37, %39 : i1
               %41 = scf.if %40 -> (i1) {
                  scf.yield %true : i1
               } else {
                  %42 = arith.cmpi eq, %35, %c99_i32 : i32
                  scf.yield %42 : i1
               }
               scf.if %41 {
                  /// Load f[i][j]
                  %42 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %43 = memref.load %42[%arg7] : memref<?xf64>
                  /// Load unew[i][j]
                  %44 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  /// Store unew[i][j] = f[i][j]
                  memref.store %43, %44[%arg7] : memref<?xf64>
               } else {
                  /// Load u[i-1][j]
                  %42 = arts.db_ref %arg4[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  /// Load u[i][j+1]
                  %43 = memref.load %42[%arg7] : memref<?xf64>
                  /// Load u[i][j-1]
                  %44 = arith.addi %arg7, %c1 : index
                  /// Load u[i+1][j]
                  %45 = arts.db_ref %arg5[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %46 = memref.load %45[%44] : memref<?xf64>
                  %47 = arith.addf %43, %46 : f64
                  %48 = arith.addi %arg7, %c-1 : index
                  %49 = memref.load %45[%48] : memref<?xf64>
                  %50 = arith.addf %47, %49 : f64
                  /// Load f[i][j]
                  %51 = arts.db_ref %arg6[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %52 = memref.load %51[%arg7] : memref<?xf64>
                  %53 = arith.addf %50, %52 : f64
                  /// Load unew[i][j]
                  %54 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %55 = memref.load %54[%arg7] : memref<?xf64>
                  %56 = arith.mulf %55, %cst_0 : f64
                  %57 = arith.mulf %56, %cst_0 : f64
                  %58 = arith.addf %53, %57 : f64
                  %59 = arith.mulf %58, %cst : f64
                  /// Store unew[i][j] = 0.25 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy)
                  %60 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  memref.store %59, %60[%arg7] : memref<?xf64>
               }
               } {...}
               arts.db_release(%arg2) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg3) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg4) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg5) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg6) : memref<?xmemref<memref<?xf64>>>
            }
         } {...}
         } {...}
      } : i64
      scf.for %arg0 = %c0 to %c100 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
         %31 = arts.db_ref %ptr_7[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         %32 = memref.load %31[%arg1] : memref<?xf64>
         memref.store %32, %alloc[%arg0, %arg1] : memref<?x100xf64>
         } {...}
      } {...}
      scf.for %arg0 = %c0 to %c100 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
         %31 = arts.db_ref %ptr_5[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %31[%arg1] : memref<?xf64>
         %32 = arts.db_ref %ptr_7[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %32[%arg1] : memref<?xf64>
         } {...}
      } {...}
      ...
      return %30 : i32
   }
   }
    ```

4. **Finally lets carts execute and check**
```bash
    carts execute poisson-task.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./poisson-task_arts
```

5. **Run with carts benchmarks and check**
```bash
    carts benchmarks run kastors-jacobi/poisson-task --trace
```

## Partitioning Pipeline Overview

The transformation from OpenMP code to ARTS datablocks happens in three stages:

| Stage | Pass | What Happens |
|-------|------|--------------|
| 7 | CreateDbs | Creates COARSE DbAllocs (sizes=[1]) with partition hints on acquires |
| 10 | ForLowering | Refines partition hints based on loop structure |
| 11 | DbPartitioning | Analyzes hints, expands multi-entry, rewrites to fine-grained |

### Key Insight: Hints vs. DB-Space

CreateDbs emits coarse DB-space (`offsets=[0], sizes=[allocSizes]`) regardless of
fine-grained intent. The intent is carried in `partition_*` hint fields:
- `partition_indices`: element coordinates [%i, %i-1, %i+1]
- `partition_offsets`: range start
- `partition_sizes`: range size

DbPartitioning consumes these hints and rewrites the DB-space fields.

### Stage-by-Stage Verification

```bash
# Stage 7: CreateDbs - coarse allocations with partition hints
carts run poisson-task.mlir --create-dbs | grep -E "partition_indices|partition_offsets"
```

**Example output (CreateDbs):**
```mlir
/// Single-entry acquire with partition hints:
arts.db_acquire[<inout>] ... partitioning(<fine_grained>, indices[%arg1]),
    offsets[%c0], sizes[%c1]
    partition_entry_modes = array<i32: 2>,       /// 2 = fine_grained
    partition_indices_segments = array<i32: 1>   /// 1 index per entry

/// Multi-entry acquire for stencil pattern (u[i-1], u[i], u[i+1]):
arts.db_acquire[<in>] ... partitioning(<fine_grained>, indices[%56, %arg1, %57]),
    offsets[%c0], sizes[%c1]
    partition_entry_modes = array<i32: 2, 2, 2>,          /// all fine_grained
    partition_indices_segments = array<i32: 1, 1, 1>      /// 1 index per entry
```

```bash
# Stage 10: ForLowering - refined partition hints
carts run poisson-task.mlir --for-lowering | grep -E "partition_"

# Stage 11: DbPartitioning - fine-grained expansion
carts run poisson-task.mlir --concurrency-opt | grep -E "partition_mode|indices\["
```

**Example output (DbPartitioning - after expansion):**
```mlir
/// Multi-entry acquire expanded into separate acquires:
/// Entry 0: u[i-1]
arts.db_acquire[<in>] ... partitioning(<fine_grained>), indices[%57],
    offsets[%c0], sizes[%c1] bounds_valid(%62)

/// Entry 1: u[i]
arts.db_acquire[<in>] ... indices[%arg1], offsets[%c0], sizes[%c1]

/// Entry 2: u[i+1]
arts.db_acquire[<in>] ... partitioning(<fine_grained>), indices[%58],
    offsets[%c0], sizes[%c1] bounds_valid(%65)
```

## Multi-Entry Expansion for Explicit Dependencies

The OpenMP depend clause:
```c
#pragma omp task depend(in: f[i], u[i-1], u[i], u[i+1]) depend(out: unew[i])
```

Creates a **multi-entry acquire** where `u` is accessed at three different indices (i-1, i, i+1) - a stencil pattern with explicit dependencies.

### How Multi-Entry Acquires Are Created

After CreateDbs, the multi-entry acquire for `u` looks like:
```mlir
arts.db_acquire[<in>] ... indices[%i_minus_1, %i, %i_plus_1]
    partition_indices = [%i_minus_1, %i, %i_plus_1]
    partition_entry_modes = [fine_grained, fine_grained, fine_grained]
```

### Phase 0: Uniformity Check

Before expansion, DbPartitioning checks if all entries have the same partition mode:
- All entries must be `fine_grained` (uniform)
- If modes differ, falls back to stencil/ESD mode

For poisson-task, all three entries (`u[i-1]`, `u[i]`, `u[i+1]`) are explicitly specified,
so they're all `fine_grained` and expansion proceeds.

### Phase 1: Create Expanded Acquires

For each entry (i-1, i, i+1), `expandMultiEntryAcquires()`:
1. Creates a separate `DbAcquireOp`
2. Extracts `partition_indices` for that entry
3. Sets single-entry partition hints

**Before expansion (multi-entry):**
```mlir
%ptr = arts.db_acquire[<in>] ... indices[%i_m1, %i, %i_p1] ...
arts.edt ... (%ptr) {
^bb0(%arg2: ...):
  // All three rows accessed via same %arg2
}
```

**After expansion (separate acquires):**
```mlir
%ptr_0 = arts.db_acquire[<in>] ... indices[%i_m1] ...   // u[i-1]
%ptr_1 = arts.db_acquire[<in>] ... indices[%i] ...     // u[i]
%ptr_2 = arts.db_acquire[<in>] ... indices[%i_p1] ...  // u[i+1]
arts.edt ... (%ptr_0, %ptr_1, %ptr_2) {
^bb0(%arg2: ..., %arg3: ..., %arg4: ...):
  // Each row accessed via its own parameter
}
```

### Phase 2: Remap DbRef Operations

After expansion, `DbRef` operations must be remapped to the correct expanded acquire.
The algorithm uses score-based matching:

| Score | Condition | Meaning |
|-------|-----------|---------|
| +2 | `accessIdx == partitionIdx` | Exact match |
| +1 | `dependsOn(accessIdx, partitionIdx)` | Computed from partition index |

**Example matching for `u[i-1]`, `u[i]`, `u[i+1]`:**
- Access `u[%i_minus_1]` → matches acquire with `partition_indices=[%i_minus_1]` (score=2)
- Access `u[%arg1]` → matches acquire with `partition_indices=[%arg1]` (score=2)
- Access `u[%i_plus_1]` → matches acquire with `partition_indices=[%i_plus_1]` (score=2)

## Index Localization for Fine-Grained Mode

For poisson-task with 2D array `u[i][j]`:

### Key Insight: partition_indices Identify What to Acquire

The `partition_indices` (like `%i-1`, `%i`, `%i+1`) are NOT used directly as db_ref indices.
Instead they:
1. **Identify which element to acquire** - passed to arts.db_acquire as `indices[]`
2. **Set up single-element ownership** - acquire sets offsets=[0], sizes=[1]
3. **Result in db_ref[0]** - because each EDT owns exactly ONE element

### The Actual Formula

```
dbRefIndex = globalIndex - partition_index
```

For single-element acquires (sizes=[1]), this ALWAYS results in [0]:
- We acquired exactly the element at `partition_index`
- So `globalIndex == partition_index` for the outer dimension
- Therefore `dbRefIndex = partition_index - partition_index = 0`

### Example: depend(in: u[i-1], u[i], u[i+1])

After expansion into 3 separate acquires:

| Acquire | partition_indices | acquire.indices | acquire.offsets | acquire.sizes |
|---------|-------------------|-----------------|-----------------|---------------|
| #1      | [%i-1]            | [%i-1]          | [%c0]           | [%c1]         |
| #2      | [%i]              | [%i]            | [%c0]           | [%c1]         |
| #3      | [%i+1]            | [%i+1]          | [%c0]           | [%c1]         |

Inside the EDT:
- `%arg2` holds row i-1 → access with `db_ref %arg2[%c0]`
- `%arg3` holds row i   → access with `db_ref %arg3[%c0]`
- `%arg4` holds row i+1 → access with `db_ref %arg4[%c0]`

The `[%c0]` is correct because each parameter holds exactly one row!

### When outerRank > 1 (Multi-Dimensional Partitioning)

For 2D partitioning with `partition_indices = [%i, %j]`:
- `dbRefIndex = [globalI - %i, globalJ - %j]`
- If single-element acquire: results in `[0, 0]`
- If range acquire (sizes > 1): results in local offset within range

**Example for 3D array with 2D partitioning:**
- Array: `A[100][50][20]` (3D)
- `partition_indices = [%i, %j]` (2 pinned dims)
- Access: `memref.load A[i][j][k]`
- Result: `dbRef[0, 0]`, `memref[%k]`

### Key Formula: outerRank = partition_indices.size()

For uniform patterns, the outer rank (number of pinned dimensions) equals the number of partition indices:

```
outerRank = partition_indices.size()  // Number of pinned dimensions
innerRank = arrayRank - outerRank     // Remaining dimensions
```

**Examples:**
- 2D array `A[i][j]` with `partition_indices = [%i]` → `outerRank = 1`, `innerRank = 1`
- 3D array `A[i][j][k]` with `partition_indices = [%i, %j]` → `outerRank = 2`, `innerRank = 1`
- 4D array with `partition_indices = [%i, %j, %k]` → `outerRank = 3`, `innerRank = 1`

## Expected Partitioning Behavior

### Explicit Dependencies -> Element-Wise (No ESD)

The poisson-task benchmark **already declares explicit dependencies**:
`depend(in : f[i], u[i - 1], u[i], u[i + 1]) depend(out : unew[i])`.
That means the stencil dependencies are *fully specified* by the user, so the
compiler should **expand** multi-entry acquires and use **fine-grained**
partitioning. ESD/stencil mode is unnecessary here.

**Expected behavior after DbPartitioning:**

1. **Multi-entry expansion (even if stencil-like)**: `expandMultiEntryAcquires()`
   should expand the multi-entry acquire into separate acquires because the
   entries are explicitly fine-grained.

2. **Partition mode stays fine-grained**:
   - `partition_mode<fine_grained>` on u/unew/f acquires
   - No `<stencil>` mode for poisson-task

3. **Debug verification**:
   ```bash
   carts run poisson-task.mlir --concurrency-opt --debug-only=db_partitioning 2>&1 | grep -E "expand|entry|multi"
   ```

   **Example debug output:**
   ```
   [DEBUG] [db_partitioning]   Found 1 multi-entry acquires to expand
   [DEBUG] [db_partitioning]   Stencil pattern detected for acquire with 3 entries - expanding for explicit fine-grained deps
   [DEBUG] [db_partitioning]   Expanding acquire with 3 entries: arts.db_acquire[<in>] ... partitioning(<fine_grained>, indices[%57, %arg1, %58])
   [DEBUG] [db_partitioning]     Created expanded acquire: arts.db_acquire[<in>] ... partitioning(<fine_grained>, indices[%57])
   ```

   Should show: `Expanding acquire with 3 entries` and **not** show
   `skipping expansion for ESD mode`.

### IR Examples: Before/After DbPartitioning

**Before (after CreateDbs, Stage 7):**

Run: `carts run poisson-task.mlir --create-dbs | grep -A2 "indices\[%5"`

```mlir
/// Multi-entry acquire for u[i-1], u[i], u[i+1] with partition hints:
%guid_16, %ptr_17 = arts.db_acquire[<in>] (%guid_8 : memref<?xi64>, %ptr_9 : memref<?xmemref<?x?xf64>>)
    partitioning(<fine_grained>, indices[%56, %arg1, %57]),  /// %56=i-1, %arg1=i, %57=i+1
    offsets[%c0], sizes[%c1]
    partition_entry_modes = array<i32: 2, 2, 2>,             /// all fine_grained
    partition_indices_segments = array<i32: 1, 1, 1>         /// 1 index per entry
```

**After (after DbPartitioning, Stage 11):**

Run: `carts run poisson-task.mlir --concurrency-opt | grep -E "db_acquire.*indices"`

```mlir
/// Multi-entry acquire expanded into 3 separate fine-grained acquires:

/// Entry 0: u[i-1] - row before current
%guid_16, %ptr_17 = arts.db_acquire[<in>] ...
    partitioning(<fine_grained>), indices[%57], offsets[%c0], sizes[%c1]
    bounds_valid(%62)   /// boundary check for i-1

/// Entry 1: u[i] - current row
%guid_18, %ptr_19 = arts.db_acquire[<in>] ...
    indices[%arg1], offsets[%c0], sizes[%c1]

/// Entry 2: u[i+1] - row after current
%guid_20, %ptr_21 = arts.db_acquire[<in>] ...
    partitioning(<fine_grained>), indices[%58], offsets[%c0], sizes[%c1]
    bounds_valid(%65)   /// boundary check for i+1
```

> Note: ESD/stencil mode is still valid for **implicit** stencil patterns
> (e.g., poisson-for) where dependencies are not explicitly enumerated.

## Architecture Notes: Indexer Specialization

### Existing Architecture

The codebase uses a factory pattern for mode-specific rewriters:
- `DbRewriter::create()` dispatches to mode-specific rewriters
- Each rewriter creates its corresponding indexer class
- Inheritance from `DbIndexerBase` provides specialization

For fine-grained mode, `DbElementWiseIndexer` handles index localization.

### Identified Gaps in N-Dimensional Handling

Investigation revealed gaps in how the indexer handles multi-dimensional pinned indices:

#### Gap 1: Hardcoded `outerRank == 1` Guards

**Location:** `DbElementWiseIndexer.cpp`, lines 82, 105, 227

```cpp
if (outerRank == 1 && hasOuterGlobals && !indices.empty()) {
  // This branch only handles 1D partitioning
  // Multi-dimensional cases never reach the general loop
}
```

**Impact:** 2D or N-D partitioning is blocked from executing the correct generalized logic.

#### Gap 2: Only First Partition Index Used

**Location:** `DbElementWiseIndexer.cpp`, lines 108, 228, 288

```cpp
Value elemCoord = indices.front();  // HARDCODED: only first dimension
for (unsigned i = 0; i < globalIndices.size(); ++i) {
  if (globalIndices[i] == elemCoord || ...) {
    outerIdx = i;
    break;  // ONLY FINDS ONE MATCH
  }
}
```

**Impact:** For `partition_indices = [%i, %j]` with access `A[i][j][k]`:
- **Expected:** `dbRef[%i, %j]`, `memref[%k]`
- **Actual:** `dbRef[%i]`, `memref[%j, %k]` (wrong)

#### Gap 3: Single Index Matching Instead of Per-Dimension

The matching logic finds ONE global index that depends on the partition index, then `break`s.
For multi-dimensional pinned indices, we need to match EACH partition index to its corresponding global index.

#### Summary of Hardcoded 1D Assumptions

| Location | Code | Problem |
|----------|------|---------|
| Line 82, 105, 227 | `outerRank == 1` guard | Blocks N-D path |
| Line 108, 228 | `indices.front()` | Only uses first partition index |
| Line 116, 257 | Single `push_back` | Only one db_ref index produced |
| Lines 233-239 | `break` after first match | Doesn't match all dimensions |

### Why poisson-task Works Despite Gaps

For poisson-task, the current implementation works because:
1. It's **1D partitioning** on outer dimension (partition_indices = [%i])
2. All entries are **fine-grained** (uniform)
3. The `outerRank == 1` path handles this case correctly
4. Multi-entry expansion works correctly for uniform modes

### Proposed Fix for N-Dimensional Cases

The core fix is to remove the `outerRank == 1` guards and use the existing general loop for ALL cases:

```cpp
// Match ALL partition indices to corresponding global indices
unsigned numPinnedDims = acquireIndices.size();  // Could be 1, 2, 3, ... N

for (unsigned p = 0; p < numPinnedDims; ++p) {
  Value partitionIdx = acquireIndices[p];

  // Find the global index that matches this partition index
  for (unsigned g = 0; g < globalIndices.size(); ++g) {
    if (globalIndices[g] == partitionIdx ||
        ValueUtils::dependsOn(globalIndices[g], partitionIdx)) {
      Value localIdx = builder.create<arith::SubIOp>(loc, globalIndices[g], partitionIdx);
      result.dbRefIndices.push_back(localIdx);
      break;
    }
  }
}

// All remaining global indices (not matched) go to memref
for (unsigned g = 0; g < globalIndices.size(); ++g) {
  if (!isMatched(g)) {
    result.memrefIndices.push_back(globalIndices[g]);
  }
}
```

This algorithm handles:
- 1D pinning: `partition_indices = [%i]` → `dbRef[0]`, `memref[%j, %k, ...]`
- 2D pinning: `partition_indices = [%i, %j]` → `dbRef[0, 0]`, `memref[%k, ...]`
- N-D pinning: `partition_indices = [%i, %j, ..., %n]` → `dbRef[0, 0, ..., 0]`, `memref[remaining]`

## Troubleshooting Wrong Results

If the ARTS checksum doesn't match OMP checksum:

1. **Check multi-entry expansion**:
   ```bash
   carts run poisson-task.mlir --concurrency-opt --debug-only=db_partitioning 2>&1 | grep -E "expand|stencil"
   ```
   You should see expansion logs and **no** ESD skip messages.

2. **Verify partition mode in IR**:
   ```bash
   carts run poisson-task.mlir --concurrency-opt | grep "partition_mode"
   ```
   Should show `<fine_grained>` for u/unew/f acquires in the computation loop.

3. **Verify fine-grained range semantics** (if offsets/sizes appear):
   In fine-grained mode, `offsets/sizes` are **element coordinates** describing
   a contiguous range, not block coordinates (no div/mod localization).

4. **Check index localization in EDT**:
   ```bash
   carts run poisson-task.mlir --concurrency-opt | grep "db_ref"
   ```
   After expansion, db_ref indices should be `[%c0]` for single-element acquires.

5. **Debug localization details**:
   ```bash
   carts run poisson-task.mlir --concurrency-opt --debug-only=db_element_wise_indexer 2>&1 | head -50
   ```

### Expected Results

After fix, expected:
- ARTS checksum: ~773.78 (matching OMP)
- Correctness: PASS
- Debug output shows: `mode=FineGrained` and expanded acquires for u[i-1], u[i], u[i+1]

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

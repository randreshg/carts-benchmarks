# jacobi-task-dep example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the jacobi-task-dep example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/kastors-jacobi/jacobi-task-dep
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist jacobi-task-dep.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> jacobi-task-dep_seq.mlir
      carts run jacobi-task-dep_seq.mlir --collect-metadata &> jacobi-task-dep_arts_metadata.mlir
      carts cgeist jacobi-task-dep.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> jacobi-task-dep.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the create-dbs pipeline
    ```bash
      carts run jacobi-task-dep.mlir --create-dbs &> jacobi-task-dep_create-dbs.mlir
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
    carts execute jacobi-task-dep.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./jacobi-task-dep_arts
```

---

## Stencil Bounds Checking

The jacobi-task-dep benchmark demonstrates a classic 5-point stencil pattern where each compute task depends on neighboring rows. This section walks through how CARTS handles out-of-bounds stencil indices.

### 1. The Problem: Out-of-Bounds Stencil Dependencies

In the original C code, the compute task has this dependency pattern:

```c
#pragma omp task depend(in: f[i], u[i-1], u[i], u[i+1]) depend(out: unew[i])
```

At iteration boundaries:
- When `i=0`: `u[i-1]` = `u[-1]` is **out of bounds**
- When `i=nx-1`: `u[i+1]` = `u[nx]` is **out of bounds**

In ARTS, EDTs are created with a fixed `numDeps` count at compile time. All slots must be satisfied for the EDT to fire. Simply skipping an invalid dependency causes **deadlock**.

### 2. Analyze the DbPass with bounds checking

Run the `--db-opt` pass to see how stencil bounds are detected:

```bash
carts run jacobi-task-dep.mlir --db-opt --debug-only=db 2>&1 | grep -E "Stencil|bounds"
```

The DbPass (`lib/arts/Passes/Db.cpp:analyzeStencilBounds()`) detects stencil patterns by analyzing index expressions. When it finds an index with non-zero offset (like `i-1` or `i+1`), it generates bounds checks.

Run the full pipeline to see the bounds_valid attribute:

```bash
carts run jacobi-task-dep.mlir --db-opt &> jacobi-task-dep_db-opt.mlir
```

Examine the output for stencil dependencies:
```mlir
scf.for %arg1 = %c0 to %100 step %c1 {
   /// Compute the stencil indices
   %idx_minus_1 = arith.addi %arg1, %c-1 : index    /// i-1
   %idx_plus_1 = arith.addi %arg1, %c1 : index      /// i+1

   /// Bounds check for u[i-1]: is (i-1) in range [0, size)?
   %ge_zero = arith.cmpi sge, %idx_minus_1, %c0 : index
   %lt_size = arith.cmpi slt, %idx_minus_1, %size : index
   %bounds_valid_minus = arith.andi %ge_zero, %lt_size : i1

   /// Bounds check for u[i+1]: is (i+1) in range [0, size)?
   %ge_zero_1 = arith.cmpi sge, %idx_plus_1, %c0 : index
   %lt_size_1 = arith.cmpi slt, %idx_plus_1, %size : index
   %bounds_valid_plus = arith.andi %ge_zero_1, %lt_size_1 : i1

   /// Acquire f[i] - no bounds_valid (always valid, index is loop IV)
   %guid_f, %ptr_f = arts.db_acquire[<in>] (%guid : ...) indices[%arg1] ... -> ...

   /// Acquire unew[i] - no bounds_valid (always valid)
   %guid_unew, %ptr_unew = arts.db_acquire[<out>] (%guid_6 : ...) indices[%arg1] ... -> ...

   /// Acquire u[i-1] - HAS bounds_valid (stencil offset -1)
   %guid_u_m1, %ptr_u_m1 = arts.db_acquire[<in>] (%guid_4 : ...) indices[%idx_minus_1] bounds_valid(%bounds_valid_minus) ... -> ...

   /// Acquire u[i] - no bounds_valid (always valid)
   %guid_u, %ptr_u = arts.db_acquire[<in>] (%guid_4 : ...) indices[%arg1] ... -> ...

   /// Acquire u[i+1] - HAS bounds_valid (stencil offset +1)
   %guid_u_p1, %ptr_u_p1 = arts.db_acquire[<in>] (%guid_4 : ...) indices[%idx_plus_1] bounds_valid(%bounds_valid_plus) ... -> ...

   /// EDT with 5 dependencies
   arts.edt <task> ... (%ptr_f, %ptr_unew, %ptr_u_m1, %ptr_u, %ptr_u_p1) : ... {
      ...
   }
}
```

### 3. Analyze EdtLowering with bounds_valid propagation

Run the `--edt-lowering` pass to see how bounds_valid is propagated:

```bash
carts run jacobi-task-dep.mlir --edt-lowering &> jacobi-task-dep_edt-lowering.mlir
```

The EdtLowering pass (`lib/arts/Passes/EdtLowering.cpp`) creates `RecordDepOp` with aligned `boundsValids` array:

```mlir
/// The boundsValids array is aligned with depGuids:
/// - boundsValids[0] = true (f[i] always valid)
/// - boundsValids[1] = true (unew[i] always valid)
/// - boundsValids[2] = %bounds_valid_minus (u[i-1] needs check)
/// - boundsValids[3] = true (u[i] always valid)
/// - boundsValids[4] = %bounds_valid_plus (u[i+1] needs check)
arts.record_dep %edtGuid deps(%guid_f, %guid_unew, %guid_u_m1, %guid_u, %guid_u_p1)
                bounds_valids(%true, %true, %bounds_valid_minus, %true, %bounds_valid_plus) ...
```

**CRITICAL**: The `boundsValids` array MUST be aligned with `depGuids`. Each `boundsValids[i]` corresponds to `depGuids[i]`. For deps without bounds checking, we use constant `true` (always valid).

### 4. Analyze ConvertArtsToLLVM with conditional signaling

Run the full pipeline to LLVM:

```bash
carts run jacobi-task-dep.mlir --arts-to-llvm &> jacobi-task-dep_arts-to-llvm.mlir
```

The ConvertArtsToLLVM pass (`lib/arts/Passes/ConvertArtsToLLVM.cpp:recordSingleDb()`) generates conditional dependency recording:

```mlir
/// For each dependency slot:
scf.for %slot = %c0 to %numDeps step %c1 {
   /// Load the boundsValid flag for this slot
   %is_valid = ... /// boundsValids[slot]

   /// Conditional dependency recording
   scf.if %is_valid {
      /// Valid bounds → normal artsRecordDep
      func.call @artsRecordDep(%dbGuid, %edtGuid, %slot, %mode)
   } else {
      /// Invalid bounds → artsSignalEdtNull to satisfy slot without data
      func.call @artsSignalEdtNull(%edtGuid, %slot)
   }
}
```

**Why `artsSignalEdtNull` instead of `artsSignalEdtValue`?**

- `artsSignalEdtValue(edt, slot, 0)` stores `0` in the GUID field with `ARTS_SINGLE_VALUE` mode. ARTS may try to route based on this "GUID", causing crashes.
- `artsSignalEdtNull(edt, slot)` uses `ARTS_NULL` mode with `NULL_GUID`. This mode simply decrements `depcNeeded` without data routing.

The `artsSignalEdtNull` function was added to the ARTS runtime (`external/arts/core/src/runtime/compute/EdtFunctions.c`):

```c
void artsSignalEdtNull(artsGuid_t edtGuid, uint32_t slot) {
  /// Uses ARTS_NULL mode - slot is satisfied without data
  internalSignalEdt(edtGuid, slot, NULL_GUID, ARTS_NULL, NULL, 0);
}
```

### 5. Runtime behavior

For the compute task at `i=0` with 5 dependencies, the runtime log shows:

```
Creating EDT[Id:1000, Guid:..., Deps:5]
Signal DB[f[0]]       → Slot 0 (artsRecordDep - valid)
Signal DB[unew[0]]    → Slot 1 (artsRecordDep - valid)
Signal NULL           → Slot 2 (artsSignalEdtNull - u[-1] is OOB!)
Signal DB[u[0]]       → Slot 3 (artsRecordDep - valid)
Signal DB[u[1]]       → Slot 4 (artsRecordDep - valid)
EDT[Id:1000] is ready (all 5 slots satisfied)
```

Inside the EDT, the kernel code already has boundary guards from the original C code:
```c
if (i == 0 || j == 0 || i == nx-1 || j == ny-1) {
  unew[i][j] = f[i][j];  /// Boundary: skip stencil computation
} else {
  unew[i][j] = 0.25 * (u[i-1][j] + ...);  /// Interior: full stencil
}
```

### 6. Pipeline summary

| Pass | File | What it does |
|------|------|--------------|
| `DbPass::analyzeStencilBounds()` | `lib/arts/Passes/Db.cpp` | Detects stencil patterns, generates bounds checks, attaches `bounds_valid` to `DbAcquireOp` |
| `EdtLowering` | `lib/arts/Passes/EdtLowering.cpp` | Propagates `bounds_valid` to `RecordDepOp`, maintains alignment with depGuids |
| `ConvertArtsToLLVM::recordSingleDb()` | `lib/arts/Passes/ConvertArtsToLLVM.cpp` | Generates `scf.if` with `artsRecordDep` / `artsSignalEdtNull` branches |

### 7. Verification

```bash
carts execute jacobi-task-dep.c -O3 -DSIZE=100
./jacobi-task-dep_arts
```

Expected output:
```
Jacobi Task-Dep
Grid size: 100 x 100
Iterations: 10
Running parallel version with task dependencies...
Running sequential version for verification...
Verification: PASS (RMS error: 0.00e+00)
```

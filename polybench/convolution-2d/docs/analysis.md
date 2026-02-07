# Convolution-2d example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the convolution-2d example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/convolution-2d
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist convolution-2d.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> convolution-2d_seq.mlir
      carts run convolution-2d_seq.mlir --collect-metadata &> convolution-2d_arts_metadata.mlir
      carts cgeist convolution-2d.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> convolution-2d.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the concurrency pipeline
    ```bash
      carts run convolution-2d.mlir --concurrency &> convolution-2d_concurrency.mlir
    ```

    Check the output of the concurrency pipeline.
    ```mlir
    module attributes {...} {
    ...
    func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
        ...
        /// Array A - Coarse grained allocation
        %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c64] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        /// Array B - Coarse grained allocation
        %guid_9, %ptr_10 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c64] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        /// Initialize array A
        scf.for %arg2 = %c0 to %c64 step %c1 {
        %7 = arith.index_cast %arg2 : index to i32
        scf.for %arg3 = %c0 to %c64 step %c1 {
            %8 = arith.index_cast %arg3 : index to i32
            %9 = arith.addi %7, %8 : i32
            %10 = arith.sitofp %9 : i32 to f32
            %11 = arith.divf %10, %cst_8 : f32
            /// Since Array A is a coarse grained allocation we use a db_ref with index 0, and then we have a single element access (the 0th element). Remember that the db_ref iterates over the outer dimensions (the sizes).
            %12 = arts.db_ref %ptr[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
            memref.store %11, %12[%arg2, %arg3] : memref<?x?xf32>
        } {...}
        } {...}
        /// Parallel Work before the arts.for loop. This will me removed later on by the edt pass in the concurrency-opt pipeline.
        %guid_11, %ptr_12 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        %guid_13, %ptr_14 = arts.db_acquire[<out>] (%guid_9 : memref<?xi64>, %ptr_10 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        arts.edt <parallel> <intranode> route(%c0_i32) (%ptr_12, %ptr_14) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> attributes {workers = #arts.workers<16>} {
        ^bb0(%arg2: memref<?xmemref<memref<?x?xf32>>>, %arg3: memref<?xmemref<memref<?x?xf32>>>):
        arts.db_release(%arg2) : memref<?xmemref<memref<?x?xf32>>>
        arts.db_release(%arg3) : memref<?xmemref<memref<?x?xf32>>>
        }
        /// Epochs are used to create the tasks and the parallel_for
        %5 = arts.epoch {
        scf.for %arg2 = %c0 to %c16 step %c1 {
            ...
            /// If the current worker has work to do, we acquire the input and output dbs and create a task.
            scf.if %13 {
            /// For work
            %guid_15, %ptr_16 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%7] size_hints[%12] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_17, %ptr_18 = arts.db_acquire[<out>] (%guid_9 : memref<?xi64>, %ptr_10 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%7] size_hints[%12] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_16, %ptr_18) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> {
            ^bb0(%arg3: memref<?xmemref<memref<?x?xf32>>>, %arg4: memref<?xmemref<memref<?x?xf32>>>):
                scf.for %arg5 = %c0 to %12 step %c1 {
                %...
                scf.for %arg6 = %c1 to %c63 step %c1 {
                    ...
                } {...}
                } {...}
                arts.db_release(%arg3) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg4) : memref<?xmemref<memref<?x?xf32>>>
            }
            }
        }
        } : i64
        ...
        arts.db_free(%guid_9) : memref<?xi64>
        arts.db_free(%ptr_10) : memref<?xmemref<memref<?x?xf32>>>
        arts.db_free(%guid) : memref<?xi64>
        arts.db_free(%ptr) : memref<?xmemref<memref<?x?xf32>>>
        return %c0_i32 : i32
    }
    ...
    }
    ```
4. **Concurrency-opt checkpoint:**
    ```bash
      carts run convolution-2d.mlir --concurrency-opt &> convolution-2d_concurrency_opt.mlir
    ```
    Check the output of the concurrency optimization pipeline.
    ```mlir
      module attributes {...} {
        ...
        func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
            ...
            /// A allocation - This is a coarse grained allocation because the access A[i][j] - since it is a stencill access, we
            /// fallback to a coarse grained allocation. This can be optimized to a fine grained allocation in the future.
            %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c64] {...} : (memref<?xi64>, memref<?xmemref<?x?xf32>>)
            /// B allocation - This is a fine grained allocation because the access B[j] doesnt depend on the 
            ///parallel loop variable i. We promoted the first element size dimension to the sizes.
            %guid_10, %ptr_11 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%4] elementType(f32) elementSizes[%c64] {...} : (memref<?xi64>, memref<?xmemref<?xf32>>)
            scf.for %arg2 = %c0 to %c64 step %c1 {
            %19 = arith.index_cast %arg2 : index to i32
            scf.for %arg3 = %c0 to %c64 step %c1 {
                /// Initialization of the A array.
                ....
                %24 = arts.db_ref %ptr[%c0] : memref<?xmemref<?x?xf32>> -> memref<?x?xf32>
                memref.store %23, %24[%arg2, %arg3] : memref<?x?xf32>
            } {...}
            } {...}
            ...
            /// for region...
            %8 = arts.epoch {
            scf.for %arg2 = %c0 to %c16 step %c1 {
                %19 = arith.muli %arg2, %c4 : index
                %20 = arith.cmpi uge, %19, %c62 : index
                %21 = arith.subi %c62, %19 : index
                %22 = arith.select %20, %c0, %21 : index
                %23 = arith.minui %22, %c4 : index
                %24 = arith.minui %23, %22 : index
                %25 = arith.cmpi ugt, %24, %c0 : index
                scf.if %25 {
                /// Acquire the A array. 
                %guid_12, %ptr_13 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?x?xf32>>) offsets[%c0] sizes[%c1] offset_hints[%19] size_hints[%24]  -> (memref<?xi64>, memref<?xmemref<?x?xf32>>)
                /// Acquire the B array.
                %guid_14, %ptr_15 = arts.db_acquire[<out>] (%guid_10 : memref<?xi64>, %ptr_11 : memref<?xmemref<?xf32>>) offsets[%19] sizes[%24] offset_hints[%19] size_hints[%24]  -> (memref<?xi64>, memref<?xmemref<?xf32>>)
                /// Create the task.
                arts.edt <task> <intranode> route(%c0_i32) (%ptr_13, %ptr_15) : memref<?xmemref<?x?xf32>>, memref<?xmemref<?xf32>> attributes {arts.id = 51 : i64} {
                ^bb0(%arg3: memref<?xmemref<?x?xf32>>, %arg4: memref<?xmemref<?xf32>>):
                    scf.for %arg5 = %c0 to %24 step %c1 {
                    /// Compute the initial we will be accessing in the B array.
                    %26 = arith.addi %19, %arg5 : index
                    %27 = arith.addi %26, %c1 : index
                    %28 = arith.index_cast %27 : index to i32
                    %29 = arith.addi %28, %c-1_i32 : i32
                    %30 = arith.index_cast %29 : i32 to index
                    %31 = arith.addi %28, %c1_i32 : i32
                    %32 = arith.index_cast %31 : i32 to index
                    scf.for %arg6 = %c1 to %c63 step %c1 {
                        %33 = arith.addi %arg6, %c-1 : index
                        /// Load the A array.
                        %34 = arts.db_ref %arg3[%c0] : memref<?xmemref<?x?xf32>> -> memref<?x?xf32>
                        /// A[i][j-1]
                        %35 = memref.load %34[%30, %33] : memref<?x?xf32>
                        %36 = arith.extf %35 : f32 to f64
                        %37 = arith.mulf %36, %cst_7 : f64
                        /// A[i][j]
                        %38 = memref.load %34[%30, %arg6] : memref<?x?xf32>
                        %39 = arith.extf %38 : f32 to f64
                        %40 = arith.mulf %39, %cst_6 : f64
                        %41 = arith.addf %37, %40 : f64
                        %42 = arith.addi %arg6, %c1 : index
                        /// A[i][j+1]
                        %43 = memref.load %34[%30, %42] : memref<?x?xf32>
                        %44 = arith.extf %43 : f32 to f64
                        %45 = arith.mulf %44, %cst_5 : f64
                        %46 = arith.addf %41, %45 : f64
                        /// A[i-1][j-1]
                        %47 = memref.load %34[%27, %33] : memref<?x?xf32>
                        %48 = arith.extf %47 : f32 to f64
                        %49 = arith.mulf %48, %cst_4 : f64
                        %50 = arith.addf %46, %49 : f64
                        /// A[i-1][j]
                        %51 = memref.load %34[%27, %arg6] : memref<?x?xf32>
                        %52 = arith.extf %51 : f32 to f64
                        %53 = arith.mulf %52, %cst_3 : f64
                        %54 = arith.addf %50, %53 : f64
                        /// A[i-1][j+1]
                        %55 = memref.load %34[%27, %42] : memref<?x?xf32>
                        %56 = arith.extf %55 : f32 to f64
                        %57 = arith.mulf %56, %cst_2 : f64
                        %58 = arith.addf %54, %57 : f64
                        /// A[i+1][j-1]
                        %59 = memref.load %34[%32, %33] : memref<?x?xf32>
                        %60 = arith.extf %59 : f32 to f64
                        %61 = arith.mulf %60, %cst_1 : f64
                        %62 = arith.addf %58, %61 : f64
                        /// A[i+1][j]
                        %63 = memref.load %34[%32, %arg6] : memref<?x?xf32>
                        %64 = arith.extf %63 : f32 to f64
                        %65 = arith.mulf %64, %cst_0 : f64
                        %66 = arith.addf %62, %65 : f64
                        /// A[i+1][j+1]
                        %67 = memref.load %34[%32, %42] : memref<?x?xf32>
                        %68 = arith.extf %67 : f32 to f64
                        %69 = arith.mulf %68, %cst : f64
                        %70 = arith.addf %66, %69 : f64
                        %71 = arith.truncf %70 : f64 to f32
                        /// Load the B[i], since we added the offset to the acquire B, we now substract it...
                        %72 = arith.subi %27, %19 : index
                        %73 = arts.db_ref %arg4[%72] : memref<?xmemref<?xf32>> -> memref<?xf32>
                        memref.store %71, %73[%arg6] : memref<?xf32>
                    } {...}
                    } {...}
                    arts.db_release(%arg3) : memref<?xmemref<?x?xf32>>
                    arts.db_release(%arg4) : memref<?xmemref<?xf32>>
                }
                }
            } {...}
            } : i64
            ...
            arts.db_free(%guid_10) : memref<?xi64>
            arts.db_free(%ptr_11) : memref<?xmemref<?xf32>>
            arts.db_free(%guid) : memref<?xi64>
            arts.db_free(%ptr) : memref<?xmemref<?x?xf32>>
            return %c0_i32 : i32
        }
        ...
        }
    ```

4. **Finally lets carts execute and check**
```bash
    carts execute convolution-2d.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./convolution-2d_arts
```
5. **Test with carts examples:**
   ```bash
   carts benchmarks run polybench/convolution-2d --trace --size=small
   ```
---

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

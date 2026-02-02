# Gemm example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the gemm example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/gemm
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist gemm.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> gemm_seq.mlir
      carts run gemm_seq.mlir --collect-metadata &> gemm_seq_metadata.txt
      carts cgeist gemm.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> gemm.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   Check for example the canonicalize memrefs pass
   ```bash
      carts run gemm.mlir --canonicalize-memrefs &> gemm_canonicalize_memrefs.mlir
   ```
   Check that the array of pointers has been rewritten to a memref with explicit dimensions.
   ```mlir
     module attributes {...} {
        ...
        func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
            ...
            /// The array of pointers has been rewritten to a memref with explicit dimensions.
            %alloc = memref.alloc(%5, %c512_3) : memref<?x?xf32>
            %c512_4 = arith.constant 512 : index
            %alloc_5 = memref.alloc(%5, %c512_4) : memref<?x?xf32>
            %c512_6 = arith.constant 512 : index
            %alloc_7 = memref.alloc(%5, %c512_6) : memref<?x?xf32>
            ...
            %13 = arith.select %11, %c0_i32, %12 : i32
            scf.if %11 {
            ...
            scf.for %arg0 = %c0 to %c512 step %c1 {
                %19 = arith.index_cast %arg0 : index to i32
                scf.for %arg1 = %c0 to %c512 step %c1 {
                ...
                //// Accesses were rewritten to use the memref directly.
                memref.store %24, %alloc[%arg0, %arg1] : memref<?x?xf32>
                }
            }
            ...
            omp.parallel   {
                omp.wsloop   schedule(static) for  (%arg0) : index = (%c0_14) to (%c512_11) step (%c1_13) {
                scf.for %arg1 = %c0_14 to %c512_11 step %c1_13 {
                    %19 = scf.for %arg2 = %c0_14 to %c512_11 step %c1_13 iter_args(%arg3 = %cst_12) -> (f32) {
                    %24 = memref.load %alloc[%arg0, %arg2] : memref<?x?xf32>
                    %25 = memref.load %alloc_5[%arg2, %arg1] : memref<?x?xf32>
                    %26 = arith.mulf %24, %25 : f32
                    %27 = arith.addf %arg3, %26 : f32
                    scf.yield %27 : f32
                    }
                    %20 = arith.mulf %cst_2, %19 : f32
                    %21 = memref.load %alloc_7[%arg0, %arg1] : memref<?x?xf32>
                    %22 = arith.mulf %cst_1, %21 : f32
                    %23 = arith.addf %20, %22 : f32
                    memref.store %23, %alloc_7[%arg0, %arg1] : memref<?x?xf32>
                }
                omp.yield
                }
                omp.terminator
            }
            ...
            }
            memref.dealloc %alloc_7 : memref<?x?xf32>
            memref.dealloc %alloc_5 : memref<?x?xf32>
            memref.dealloc %alloc : memref<?x?xf32>
            return %13 : i32
        }
    }
    ```
   

4. **Then check the create dbs output**
   ```bash
      carts run gemm.mlir --create-dbs &> gemm_create_dbs.mlir
   ```
   Check the inline comments that I left in the code.
   ```mlir
      [0mmodule attributes {...} {
        ....
        func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
            ...
            /// This is very important, noticed that the static size is now part of the output memref type and only one size remained in the memref.alloc
            %alloc = memref.alloc(%5) {...} : memref<?x512xf32>
            %6 = arts.db_dim %alloc, %c0 : memref<?x512xf32> -> index
            /// The DbAlloc has the dimensions as expected...
            %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%6, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %alloc_2 = memref.alloc(%5) {...} : memref<?x512xf32>
            %7 = arts.db_dim %alloc_2, %c0 : memref<?x512xf32> -> index
            %guid_3, %ptr_4 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%7, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %alloc_5 = memref.alloc(%5) {...} : memref<?x512xf32>
            %8 = arts.db_dim %alloc_5, %c0 : memref<?x512xf32> -> index
            %guid_6, %ptr_7 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%8, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            ...
            %16 = arith.select %14, %c0_i32, %15 : i32
            scf.if %14 {
            scf.for %arg0 = %c0 to %c512 step %c1 {
                %22 = arith.index_cast %arg0 : index to i32
                scf.for %arg1 = %c0 to %c512 step %c1 {
                %23 = arith.index_cast %arg1 : index to i32
                %24 = arith.addi %22, %23 : i32
                %25 = arith.remsi %24, %c13_i32 : i32
                %26 = arith.sitofp %25 : i32 to f32
                %27 = arith.mulf %26, %cst : f32
                /// Notice that since the dballoc is coarse grained, we use a db_ref with index 0, and then we have a single element access (the 0th element). Remember that the db_ref iterates over the outer dimensions (the sizes).
                %28 = arts.db_ref %ptr[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                memref.store %27, %28[%arg0, %arg1] : memref<?x?xf32>
                } {...}
            } {...}
            ...
            %guid_8, %ptr_9 = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_10, %ptr_11 = arts.db_acquire[<inout>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_12, %ptr_13 = arts.db_acquire[<inout>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1]  -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            arts.edt <parallel> <internode> route(%c0_i32) (%ptr_9, %ptr_11, %ptr_13) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> {
            ^bb0(%arg0: memref<?xmemref<memref<?x?xf32>>>, %arg1: memref<?xmemref<memref<?x?xf32>>>, %arg2: memref<?xmemref<memref<?x?xf32>>>):
                arts.for(%c0) to(%c512) step(%c1) schedule(<static>) {{
                ^bb0(%arg3: index):
                scf.for %arg4 = %c0 to %c512 step %c1 {
                    %22 = scf.for %arg5 = %c0 to %c512 step %c1 iter_args(%arg6 = %cst_1) -> (f32) {
                    %27 = arts.db_ref %arg0[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                    %28 = memref.load %27[%arg3, %arg5] : memref<?x?xf32>
                    %29 = arts.db_ref %arg1[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                    %30 = memref.load %29[%arg5, %arg4] : memref<?x?xf32>
                    %31 = arith.mulf %28, %30 : f32
                    %32 = arith.addf %arg6, %31 : f32
                    scf.yield %32 : f32
                    } {...}
                    %23 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                    %24 = memref.load %23[%arg3, %arg4] : memref<?x?xf32>
                    %25 = arith.mulf %24, %cst_1 : f32
                    %26 = arith.addf %22, %25 : f32
                    memref.store %26, %23[%arg3, %arg4] : memref<?x?xf32>
                } {...}
                }} {...}
                arts.db_release(%arg0) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg1) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg2) : memref<?xmemref<memref<?x?xf32>>>
            }
            ...
            }
            arts.db_free(%guid_6) : memref<?xi64>
            arts.db_free(%ptr_7) : memref<?xmemref<memref<?x?xf32>>>
            arts.db_free(%guid_3) : memref<?xi64>
            arts.db_free(%ptr_4) : memref<?xmemref<memref<?x?xf32>>>
            arts.db_free(%guid) : memref<?xi64>
            arts.db_free(%ptr) : memref<?xmemref<memref<?x?xf32>>>
            return %16 : i32
        }
        }








4. **Then check the concurrency output**
   ```bash
      carts run gemm.mlir --concurrency &> gemm_concurrency.mlir
   ```
   Analyze the inline comments that I left in the code.
   I will provide a summarized version of the module after all finished. 
   ```mlir
    module attributes {...} {
        ....
        func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
            ...
            %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%5, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_3, %ptr_4 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%5, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_6, %ptr_7 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%5, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            ...
        
            scf.if %11 {
            ...
            /// Parallel work before the arts.for loop. This will me removed later on by the edt pass in the concurrency-opt pipeline.
            %guid_8, %ptr_9 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_10, %ptr_11 = arts.db_acquire[<in>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_12, %ptr_13 = arts.db_acquire[<inout>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            arts.edt <parallel> <intranode> route(%c0_i32) (%ptr_9, %ptr_11, %ptr_13) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> attributes {workers = #arts.workers<16>} {
            ^bb0(%arg0: memref<?xmemref<memref<?x?xf32>>>, %arg1: memref<?xmemref<memref<?x?xf32>>>, %arg2: memref<?xmemref<memref<?x?xf32>>>):
                arts.db_release(%arg0) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg1) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg2) : memref<?xmemref<memref<?x?xf32>>>
            }
            %14 = arts.epoch {
                scf.for %arg0 = %c0 to %c16 step %c1 {
                ...
                /// If the current worker has work to do, we perform he work
                scf.if %26 {
                    %guid_14, %ptr_15 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%20] size_hints[%25] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
                    %guid_16, %ptr_17 = arts.db_acquire[<in>] (%guid_3 : memref<?xi64>, %ptr_4 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%20] size_hints[%25] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
                    %guid_18, %ptr_19 = arts.db_acquire[<inout>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%20] size_hints[%25] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
                    arts.edt <task> <intranode> route(%c0_i32) (%ptr_15, %ptr_17, %ptr_19) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> {
                    ^bb0(%arg1: memref<?xmemref<memref<?x?xf32>>>, %arg2: memref<?xmemref<memref<?x?xf32>>>, %arg3: memref<?xmemref<memref<?x?xf32>>>):
                    scf.for %arg4 = %c0 to %25 step %c1 {
                        %27 = arith.addi %20, %arg4 : index
                        scf.for %arg5 = %c0 to %c512 step %c1 {
                        %28 = scf.for %arg6 = %c0 to %c512 step %c1 iter_args(%arg7 = %cst_1) -> (f32) {
                            %33 = arts.db_ref %arg1[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                            %34 = memref.load %33[%27, %arg6] : memref<?x?xf32>
                            %35 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                            %36 = memref.load %35[%arg6, %arg5] : memref<?x?xf32>
                            %37 = arith.mulf %34, %36 : f32
                            %38 = arith.addf %arg7, %37 : f32
                            scf.yield %38 : f32
                        } {...}
                        %29 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                        %30 = memref.load %29[%27, %arg5] : memref<?x?xf32>
                        %31 = arith.mulf %30, %cst_1 : f32
                        %32 = arith.addf %28, %31 : f32
                        memref.store %32, %29[%27, %arg5] : memref<?x?xf32>
                        } {...}
                    } {...}
                    arts.db_release(%arg1) : memref<?xmemref<memref<?x?xf32>>>
                    arts.db_release(%arg2) : memref<?xmemref<memref<?x?xf32>>>
                    arts.db_release(%arg3) : memref<?xmemref<memref<?x?xf32>>>
                    }
                }
                }
            } : i64
            ...
            }
            arts.db_free(%guid_6) : memref<?xi64>
            arts.db_free(%ptr_7) : memref<?xmemref<memref<?x?xf32>>>
            arts.db_free(%guid) : memref<?xi64>
            arts.db_free(%ptr) : memref<?xmemref<memref<?x?xf32>>>
            arts.db_free(%guid_3) : memref<?xi64>
            arts.db_free(%ptr_4) : memref<?xmemref<memref<?x?xf32>>>
            return %13 : i32
        }
        }
   ```

4. **Then check the concurrency-opt output**
    ```bash
      carts run gemm.mlir --concurrency-opt &> gemm_concurrency_opt.mlir
    ```
    This is where the partitioning happens.
    ```mlir
    module attributes {...} {
    ...
    func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
        ...
        /// Noticed that we promoted one dimension of the allocation to the element sizes.
        /// This is Array A
        %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>] route(%c0_i32 : i32) sizes[%4] elementType(f32) elementSizes[%c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf32>>>)
        /// This is Array B - Noticed that this one remained coarse grained because the Access B[k] doesnt depen on the parallel loop variable i
        %guid_2, %ptr_3 = arts.db_alloc[<in>, <heap>, <read>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        /// This is Array C - We promoted one dimension of the allocation to the element sizes.
        %guid_4, %ptr_5 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%4] elementType(f32) elementSizes[%c512] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf32>>>)
        scf.for %arg0 = %c0 to %c512 step %c1 {
        %11 = arith.index_cast %arg0 : index to i32
        scf.for %arg1 = %c0 to %c512 step %c1 {
            ...
            /// The first dimension of the db_ref is the outer dimension of the allocation.
            /// This is perfect
            %17 = arts.db_ref %ptr[%arg0] : memref<?xmemref<memref<?xf32>>> -> memref<?xf32>
            memref.store %16, %17[%arg1] : memref<?xf32>
        } {...}
        } {...}
        scf.for %arg0 = %c0 to %c512 step %c1 {
        %11 = arith.index_cast %arg0 : index to i32
        %12 = arith.muli %11, %c3_i32 : i32
        scf.for %arg1 = %c0 to %c512 step %c1 {
            /// Since this is a coarse grained allocation we use a db_ref with index 0, and then we iterate over the inner dimension (arg1)
            %18 = arts.db_ref %ptr_3[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
            memref.store %17, %18[%arg0, %arg1] : memref<?x?xf32>
        } {...}
        } {...}
        ...
        /// Parallel for epoch
        %5 = arts.epoch {
        scf.for %arg0 = %c0 to %c16 step %c1 {
            ...
            /// if the current worker has work to do, we acquire the dbs and create a task.
            scf.if %17 {
            /// Acquire A
            %guid_6, %ptr_7 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?xf32>>>) offsets[%11] sizes[%16] offset_hints[%11] size_hints[%16]  -> (memref<?xi64>, memref<?xmemref<memref<?xf32>>>)
            /// Acquire B - we didnt propagate the chunking to this one because the Access B[k] doesnt depend on the parallel loop variable i
            %guid_8, %ptr_9 = arts.db_acquire[<in>] (%guid_2 : memref<?xi64>, %ptr_3 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%11] size_hints[%16]  -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            /// Acquire C
            %guid_10, %ptr_11 = arts.db_acquire[<inout>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf32>>>) offsets[%11] sizes[%16] offset_hints[%11] size_hints[%16]  -> (memref<?xi64>, memref<?xmemref<memref<?xf32>>>)
            /// Create the task
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_7, %ptr_9, %ptr_11) : memref<?xmemref<memref<?xf32>>>, memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?xf32>>> attributes {arts.id = 33 : i64} {
            ^bb0(%arg1: memref<?xmemref<memref<?xf32>>>, %arg2: memref<?xmemref<memref<?x?xf32>>>, %arg3: memref<?xmemref<memref<?xf32>>>):
                scf.for %arg4 = %c0 to %16 step %c1 {
                scf.for %arg5 = %c0 to %c512 step %c1 {
                    %18 = scf.for %arg6 = %c0 to %c512 step %c1 iter_args(%arg7 = %cst_1) -> (f32) {
                    %23 = arts.db_ref %arg1[%arg4] : memref<?xmemref<memref<?xf32>>> -> memref<?xf32>
                    %24 = memref.load %23[%arg6] : memref<?xf32>
                    %25 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                    %26 = memref.load %25[%arg6, %arg5] : memref<?x?xf32>
                    %27 = arith.mulf %24, %26 : f32
                    %28 = arith.addf %arg7, %27 : f32
                    scf.yield %28 : f32
                    } {...}
                    %19 = arts.db_ref %arg3[%arg4] : memref<?xmemref<memref<?xf32>>> -> memref<?xf32>
                    %20 = memref.load %19[%arg5] : memref<?xf32>
                    %21 = arith.mulf %20, %cst_1 : f32
                    %22 = arith.addf %18, %21 : f32
                    memref.store %22, %19[%arg5] : memref<?xf32>
                } {...}
                } {...}
                arts.db_release(%arg1) : memref<?xmemref<memref<?xf32>>>
                arts.db_release(%arg2) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg3) : memref<?xmemref<memref<?xf32>>>
            }
            }
        } {...}
        } : i64
        ...
        return %c0_i32 : i32
    }
    }
    ```

5. **Finally lets carts execute and check**
```bash
    carts execute gemm.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./gemm_arts
```

# lulesh example analysis

Walk through these steps and fix any problem that you find in the way

---

## Bug Fix: ARTS crash with indirect indices (Fixed)

**Problem:** `lulesh_arts` crashed in `__arts_edt_3` with a null dependency
pointer.

**Root Cause:** dbpartitioning allowed chunked acquires when the index was
derived from a memory load (indirect gather via `nodelist`). The dependency
indexing then referenced depv slots that were never acquired.

**Fix:** Treat indirect indices (values derived from `memref.load`/`llvm.load`)
as non-partitionable by default. This keeps arrays like `x/y/z` coarse even
when offset/size hints exist. Use `--partition-fallback=fine` to allow
element-wise partitioning for non-affine accesses when you want to explore
performance tradeoffs.

**Result:** `carts benchmarks run lulesh --size small` passes (ARTS+OMP
verification succeeds).

---

1. **Navigate to the lulesh example directory:**

   ```bash
   cd /Users/randreshg/Documents/carts/external/carts-benchmarks/lulesh
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no `lulesh.mlir` run:

   ```bash
   carts cgeist lulesh.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> lulesh_seq.mlir
   carts run lulesh_seq.mlir --collect-metadata &> lulesh_arts_metadata.mlir
   carts cgeist lulesh.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> lulesh.mlir
   ```

3. **Run the pipeline and stop after any stage**
   Run the pipeline and stop after any stage.

   For example, check canonicalize-memrefs:
   ```bash
   carts run lulesh.mlir --canonicalize-memrefs &> lulesh_canonicalize_memrefs.mlir
   ```
   Check that array-of-arrays are rewritten to explicit memref dimensions:
   ```mlir
   // nodelist: Index_t** -> memref<?x?xi32> (outer = element, inner = 8)
   %nodelist = memref.alloc(%numElem, %c8) : memref<?x?xi32>
   ```

   For example, analyze the create-dbs pipeline:
   ```bash
   carts run lulesh.mlir --create-dbs &> lulesh_create_dbs.mlir
   ```
   Check that `arts.db_alloc` uses outer dims as `sizes[...]` and inner dims
   in `elementSizes[...]`, and that `arts.db_ref` indexes the outer dimension
   before accessing the inner memref.

   If you need to inspect initialization values, enable the debug prints:
   ```bash
   CARTS_LULESH_DEBUG=1 ./lulesh_arts -s 3 -i 1
   ```
   This prints `e/p/q/v/volo/nodalMass` and `nodelist` values after init.

4. **Concurrency-opt checkpoint**
   ```bash
   carts run lulesh.mlir --concurrency-opt &> lulesh_concurrency_opt.mlir
   ```
   Check that arrays tied to the parallel loop are chunked only when the
   access is direct. For indirect gathers (e.g., `x/y/z` indexed by
   `nodelist`), `arts.partition` should be `none` (coarse) and `db_acquire`
   offsets should be `%c0` by default. To experiment with element-wise fallback:
   ```bash
   carts run lulesh.mlir --concurrency-opt --partition-fallback=fine \
     &> lulesh_concurrency_opt_fine.mlir
   ```

5. **Finally lets carts execute and check**

   ```bash
   carts execute lulesh.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   artsConfig=arts.cfg ./lulesh_arts -s 3 -i 5
   ```

6. **Run with carts benchmarks and check**

   ```bash
   carts benchmarks run lulesh --size small
   ```

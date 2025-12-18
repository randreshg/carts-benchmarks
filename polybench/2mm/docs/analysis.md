# 2mm example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the 2mm example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/2mm
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist 2mm.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> 2mm_seq.mlir
      carts run 2mm_seq.mlir --collect-metadata &> 2mm_arts_metadata.mlir
      carts cgeist 2mm.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> 2mm.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the concurrency pipeline
    ```bash
      carts run 2mm.mlir --concurrency &> 2mm_concurrency.mlir
    ```


4. **Concurrency-opt checkpoint:**
    ```bash
      carts run 2mm.mlir --concurrency-opt &> 2mm_concurrency_opt.mlir
    ```

5. **Finally lets carts execute and check**
```bash
    carts execute 2mm.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./2mm_arts
```

6. **Run it with carts example and check**

```bash
    carts benchmarks run polybench/2mm
```
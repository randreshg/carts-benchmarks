# fdtd-2d example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the fdtd-2d example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/fdtd-2d
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist fdtd-2d.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> fdtd-2d_seq.mlir
      carts compile fdtd-2d_seq.mlir --collect-metadata &> fdtd-2d_arts_metadata.mlir
      carts cgeist fdtd-2d.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> fdtd-2d.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the concurrency pipeline
    ```bash
      carts compile fdtd-2d.mlir --concurrency &> fdtd-2d_concurrency.mlir
    ```
    Notice that it says that no arts.for operationn were found. This is an error...
    fix it. the arts.for might be in a nested region
   
4. **Concurrency-opt checkpoint:**
    ```bash
      carts compile fdtd-2d.mlir --concurrency-opt &> fdtd-2d_concurrency_opt.mlir
    ```

4. **Finally lets carts compile and check**
```bash
    carts compile fdtd-2d.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./fdtd-2d_arts
```

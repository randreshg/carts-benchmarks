# Correlation example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the correlation example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/correlation
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist correlation.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> correlation_seq.mlir
      carts compile correlation_seq.mlir --collect-metadata &> correlation__arts_metadata.mlir
      carts cgeist correlation.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> correlation.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the concurrency pipeline
    ```bash
      carts compile correlation.mlir --concurrency &> correlation_concurrency.mlir
    ```
4. **Concurrency-opt checkpoint:**
    ```bash
      carts compile correlation.mlir --concurrency-opt &> correlation_concurrency_opt.mlir
    ```

4. **Finally lets carts compile and check**
```bash
    carts compile correlation.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./correlation_arts
```

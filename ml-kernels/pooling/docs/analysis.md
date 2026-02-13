# pooling example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the pooling example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/ml-kernels/pooling
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist pooling.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> pooling_seq.mlir
      carts compile pooling_seq.mlir --collect-metadata &> pooling_arts_metadata.mlir
      carts cgeist pooling.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities > pooling.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the create-dbs pipeline
    ```bash
      carts compile pooling.mlir --create-dbs &> pooling_create-dbs.mlir
    ```

4. **Finally lets carts compile and check**
```bash
    carts compile pooling.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./pooling_arts
```

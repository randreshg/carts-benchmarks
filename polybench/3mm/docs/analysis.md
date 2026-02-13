# 3mm Benchmark Analysis

## Quick Reference

```bash
# Navigate to benchmark
cd /opt/carts/external/carts-benchmarks/polybench/3mm

# Build if needed
carts build

# Generate MLIR from source
carts cgeist 3mm.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities > 3mm_seq.mlir 2>&1
carts compile 3mm_seq.mlir --collect-metadata > 3mm_arts_metadata.mlir 2>&1
carts cgeist 3mm.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities > 3mm.mlir 2>&1

# Run full pipeline
carts compile 3mm.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
./3mm_arts

# Run benchmarks
carts benchmarks run polybench/3mm --trace --size=large
```

---

## Performance Results

| Benchmark | ARTS | OMP | Speedup | Vector Instrs |
|-----------|------|-----|---------|---------------|
| 2mm | 1.56s | 2.01s | **1.29x** | 418 |
| **3mm** | **8.60s** | **3.57s** | **0.42x** | 20 |

**Note:** 3mm performance is extremely sensitive to matmul loop shape. Without
cache/SIMD-friendly transforms, ARTS can be significantly slower than OpenMP.

---

## Update: Matmul Loop Transforms (Stage 6)

ARTS now has a Stage 6 matmul-focused pass (`arts-loop-transforms`) that:
- Rewrites SSA dot-product matmuls into an update form (`k` outer / `j` inner) with `j` tiling.
- Recognizes the post-loop-reordering update form (common in 3mm) and **hoists `A[i,k]` out of the inner `j` loop**, optionally tiling `j`.

### Confirm it triggers

```bash
# Stop after Stage 6 and show both passes
carts compile 3mm.mlir --loop-reordering --debug-only=loop_reordering,loop_transforms 2>&1
```

---

## Kernel Structure

```c
// 3mm performs 3 matrix multiplications:
static void kernel_3mm(...) {
#pragma omp parallel
  {
    // Stage 1: E = A * B
    #pragma omp for
    for (i = 0; i < NI; i++)
      for (j = 0; j < NJ; j++) {
        E[i][j] = 0;
        for (k = 0; k < NK; ++k)
          E[i][j] += A[i][k] * B[k][j];  // B accessed by k!
      }

    // Stage 2: F = C * D
    #pragma omp for
    for (i = 0; i < NJ; i++)
      for (j = 0; j < NL; j++) {
        F[i][j] = 0;
        for (k = 0; k < NM; ++k)
          F[i][j] += C[i][k] * D[k][j];  // D accessed by k!
      }

    // Stage 3: G = E * F (depends on stages 1 & 2)
    #pragma omp for
    for (i = 0; i < NI; i++)
      for (j = 0; j < NL; j++) {
        G[i][j] = 0;
        for (k = 0; k < NJ; ++k)
          G[i][j] += E[i][k] * F[k][j];  // F accessed by k!
      }
  }
}
```

---

## Root Cause: DB Partitioning Fails

### The Problem

ARTS partitions parallel loops by the outer loop variable `i`. But in matrix multiplication:
- `E[i][j]`, `A[i][k]` - first index IS `i` (partitionable)
- `B[k][j]` - first index is `k` NOT `i` (NOT partitionable!)

### Debug Output

```bash
carts compile 3mm.mlir --concurrency-opt --debug-only=db 2>&1 | grep -E "SKIP|PASS|Promoting"
```

**Results:**
```
Checking allocation: E (3mm.c:102)
  Computed chunk info... PASS
  Promoting alloc with 2 acquires, chunkSize=125

Checking allocation: F (3mm.c:105)
  SKIP: canBePartitioned() returned false

Checking allocation: G (3mm.c:108)
  Computed chunk info... PASS
  Promoting alloc with 1 acquires, chunkSize=125

Checking allocation: A (3mm.c:103)
  SKIP: H1 heuristic - read-only on single-node, keep coarse

Checking allocation: B (3mm.c:104)
  SKIP: first index of memory access not derived from partition offset

Checking allocation: C (3mm.c:106)
  SKIP: H1 heuristic - read-only on single-node, keep coarse

Checking allocation: D (3mm.c:107)
  SKIP: first index of memory access not derived from partition offset
```

### Partitioning Summary

| DB | Source Line | Status | Reason |
|----|-------------|--------|--------|
| E | 3mm.c:102 | PROMOTED | chunkSize=125, 2 acquires |
| G | 3mm.c:108 | PROMOTED | chunkSize=125, 1 acquire |
| F | 3mm.c:105 | SKIP | `canBePartitioned() false` (used by k) |
| A | 3mm.c:103 | SKIP | H1: read-only single-node |
| B | 3mm.c:104 | SKIP | Access pattern: `B[k][j]` not `B[i][...]` |
| C | 3mm.c:106 | SKIP | H1: read-only single-node |
| D | 3mm.c:107 | SKIP | Access pattern: `D[k][j]` not `D[i][...]` |

---

## Pipeline Analysis

### Stage-by-Stage Commands

```bash
# Check DB creation
carts compile 3mm.mlir --create-dbs > 3mm_dbs.mlir 2>&1

# Check concurrency
carts compile 3mm.mlir --concurrency > 3mm_conc.mlir 2>&1

# Check DB optimization (partitioning happens here)
carts compile 3mm.mlir --db-opt --debug-only=db 2>&1 | head -100

# Check after LLVM lowering
carts compile 3mm.mlir --arts-to-llvm > 3mm_arts_llvm.mlir 2>&1

# Generate LLVM IR
carts compile 3mm.mlir --emit-llvm > 3mm.ll 2>&1
```

### EDT Structure

```bash
grep -E "arts\.edt|artsEdtCreate" 3mm-arts.ll
```

**Output:**
```
define void @__arts_edt_3(...)  // E = A*B
define void @__arts_edt_2(...)  // F = C*D
define void @__arts_edt_1(...)  // G = E*F
call i64 @artsEdtCreateWithEpochArtsId(ptr @__arts_edt_1, ...)
call i64 @artsEdtCreateWithEpochArtsId(ptr @__arts_edt_2, ...)
call i64 @artsEdtCreateWithEpochArtsId(ptr @__arts_edt_3, ...)
```

---

## Vectorization Analysis

```bash
# Check vectorization hints
carts compile 3mm.mlir --emit-llvm --debug-only=arts_loop_vectorization_hints 2>&1 | grep -E "Processing|Attached|Total"

# Count vector instructions in binary
objdump -d 3mm_arts | grep -c "fmul.*v.*\.2d"
```

**Results:**
- 9 loop backedges annotated with vectorization hints
- Only 20 NEON vector instructions (vs 418 in 2mm)
- Limited vectorization due to access patterns

---

## Why 2mm is Faster

| Factor | 2mm | 3mm |
|--------|-----|-----|
| Matrix multiplies | 2 | 3 |
| EDTs | 2 | 3 |
| Inter-EDT dependencies | 1 (D depends on tmp) | 2 (G depends on E AND F) |
| Datablocks | 5 | 7 |
| Partitioned DBs | E (promoted) | E, G (promoted) |
| Vector instructions | 418 | 20 |

**Key Difference:** 3mm has more dependencies and less opportunity for parallelism.

---

## Potential Optimizations

### 1. Matrix Transposition
Transpose B, D, F before multiplication to make access `B_T[j][k]` instead of `B[k][j]`.

### 2. Loop Interchange
Change iteration order to make column accesses contiguous.

### 3. Block/Tile Partitioning
Partition both rows AND columns (2D tiling) instead of just rows.

### 4. Reduce Task Overhead
Create fewer, larger EDTs to amortize runtime overhead.

### 5. Stage Fusion
Fuse stages 1 & 2 (E=A*B and F=C*D are independent) into a single EDT.

---

## Debug Commands Reference

```bash
# Full debug output for DB partitioning
carts compile 3mm.mlir --concurrency-opt --debug-only=db 2>&1 | tee 3mm_db_debug.log

# Acquire-level partitioning debug
carts compile 3mm.mlir --concurrency-opt --debug-only=db_acquire_node 2>&1 | grep -E "Skipping|PASS"

# Check loop vectorization hints
carts compile 3mm.mlir --emit-llvm --debug-only=arts_loop_vectorization_hints 2>&1

# Full pipeline with all debug
carts compile 3mm.mlir --O3 --emit-llvm --debug 2>&1 | tee 3mm_full_debug.log
```

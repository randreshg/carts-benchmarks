# Strassen-Task - Untied Task-Based Strassen Algorithm

## Description

Strassen's fast matrix multiplication algorithm using OpenMP untied tasks with implicit synchronization. This variant implements the recursive divide-and-conquer Strassen algorithm with task parallelism controlled by cutoff depth, using taskwait barriers for synchronization between computation phases. Reduces arithmetic complexity from O(n³) to O(n^2.807).

## Algorithm

Strassen's algorithm divides n×n matrices into 2×2 blocks and computes 7 products instead of 8:

```
Traditional block multiplication: 8 products
Strassen: 7 products with clever combinations

Step 1: Compute 8 temporary matrices (S1-S8):
  S1 = A21 + A22        S5 = B12 - B
  S2 = S1 - A           S6 = B22 - S5
  S3 = A - A21          S7 = B22 - B12
  S4 = A12 - S2         S8 = S6 - B21

Step 2: Compute 7 products (M1-M7) recursively:
  M1 = (A11 + A22) * (B11 + B22)  [not stored separately]
  M2 = A * B
  M3 = S2 * S6
  M4 = S3 * S7
  M5 = S1 * S5
  M6 = S4 * B22
  M7 = A22 * S8

Step 3: Combine to form result:
  C11 = M2 + (M3 + M5 + ...)
  C12 = M6 + M3 + M5 + M2
  C21 = M7 + M4 + M3 + M2
  C22 = M5 + M3 + M2 - M7

Base case: MatrixSize <= cutoff_size → standard multiplication
```

### Parallelization Strategy

Uses `#pragma omp task untied` with taskwait synchronization:
- Creates tasks for computing temporary matrices (S1-S8)
- Creates tasks for 7 recursive multiplications
- Creates tasks for final combination phase
- Taskwait ensures each phase completes before next begins
- Untied allows task migration for better load balancing

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | N=64, cutoff=16, depth=2 | Small matrices |
| **MEDIUM** | N=128, cutoff=32, depth=2 | Medium matrices (default) |
| **LARGE** | N=256, cutoff=32, depth=3 | Large matrices |

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (N=64)
make small

# Build medium size (N=128) - default
make medium

# Build large size (N=256)
make large

# Build all pipeline stages
make all
```

### Build individual stages

```bash
# Generate sequential MLIR
make seq

# Collect runtime metadata
make metadata

# Generate parallel MLIR
make parallel

# Run concurrency analysis
make concurrency

# Run optimized concurrency analysis
make concurrency-opt
```

### Clean build artifacts

```bash
make clean
```

## Parallelization Strategy

**OpenMP untied task approach:**

```c
if (Depth < cutoff_depth) {
  // Phase 1: Compute S1-S8 (temporary matrices) with tasks
  #pragma omp task private(Row,Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      for (Column = 0; Column < QuadrantSize; Column++)
        S1[...] = A21[...] + A22[...];
  // ... S2-S8 similar ...
  #pragma omp taskwait

  // Phase 2: 7 recursive multiplications with tasks
  #pragma omp task untied
    OptimizedStrassenMultiply_par(M2, A, B, ...);  // M2 = A * B

  #pragma omp task untied
    OptimizedStrassenMultiply_par(M5, S1, S5, ...);  // M5 = S1 * S5

  // ... M3, M4, M6, M7, C11, C12, C21 similar ...
  #pragma omp taskwait

  // Phase 3: Combine results with tasks
  #pragma omp task private(Row,Column)
    for (Row = 0; Row < QuadrantSize; Row++)
      C[...] += M2[...];
  // ... C12, C21, C22 similar ...
  #pragma omp taskwait
} else {
  // Sequential computation (depth limit reached)
  // ... compute S1-S8, M2, M5, etc. without tasks ...
}
```

**Task Structure:**
- **Level 1 (Depth < cutoff_depth)**: Creates tasks for parallelism
  - 8 tasks for S1-S8 (matrix additions/subtractions)
  - 7 tasks for M1-M7 (recursive multiplications)
  - 4 tasks for combining results into C11-C22
  - Total: ~19 tasks per level
- **Level 2+**: Each M task recursively creates more tasks (if depth allows)
- **Base level (Depth >= cutoff_depth)**: Sequential execution

**Synchronization Points:**
1. After S1-S8 computed (needed for M computations)
2. After M1-M7 computed (needed for C combinations)
3. After C combinations (ensure completion)

**Key Features:**
- **Untied tasks**: Can migrate between threads for load balancing
- **Explicit barriers**: Taskwait ensures correct execution order
- **Depth control**: Limits task creation to avoid overhead
- **Recursive**: Each multiplication can spawn more tasks

**Advantages:**
- Simple synchronization model (explicit taskwait)
- Untied allows flexible scheduling
- Clear phase separation
- Controllable parallelism via cutoff_depth

**Trade-offs:**
- Taskwait may limit pipelining opportunities
- More synchronization than dependency version
- Higher overhead than task-dep at high depths

## Comparison with Other Variants

| Aspect | strassen-task | strassen-task-dep |
|--------|--------------|-------------------|
| **Dependencies** | Implicit (taskwait) | Explicit (depend clauses) |
| **Synchronization** | Taskwait barriers | Automatic data-flow |
| **Task type** | Untied | Default (with depend) |
| **Phases** | 3 phases with barriers | Data-flow driven |
| **Complexity** | Simpler | More complex |
| **Pipelining** | Limited | Natural |
| **OpenMP version** | 3.0+ (untied) | 4.0+ (depend) |

## Strassen Algorithm Mathematics

**Complexity:**
- Traditional: O(n³) arithmetic operations
- Strassen: O(n^log₂(7)) ≈ O(n^2.807)
- Crossover point: Typically n=64-128

**Why 7 products?**
- Exploits algebraic identities to reduce multiplications
- Trades multiplications (expensive) for additions (cheaper)
- Each level: 7 recursive calls instead of 8

**Memory:**
- 11 temporary matrices per level (S1-S8, M2, M5, T1sMULT)
- Memory = O(n²) * depth
- Can be significant for deep recursion

## Performance Characteristics

- **Compute**: O(n^2.807) vs O(n³)
- **Memory**: Higher than standard (temporary matrices)
- **Parallelism**: Exponential with depth (7^depth tasks)
- **Overhead**: Task creation and temporary allocation
- **Crossover**: Typically faster for n>128

## CARTS Compatibility

- No global variables
- Matrices passed as parameters
- OpenMP untied tasks
- Clean recursive structure
- Divide-and-conquer pattern
- Controllable parallelism (cutoff_depth)
- Dynamic memory management (malloc/free)

# SparseLU-Task - Untied Task-Based Sparse LU Factorization

## Description

Block-based sparse LU factorization using OpenMP untied tasks with implicit dependencies. This variant uses fork-join task parallelism with taskwait barriers to synchronize between algorithmic phases, providing a simpler task model than explicit dependencies while maintaining parallelism.

## Algorithm

Block-based LU factorization for sparse matrices using four kernel operations:

```
For each diagonal block k:
  1. LU0: Factor diagonal block A[k][k] = L[k][k] * U[k][k]

  2. FWD (Forward): For each block A[k][j] in row k:
     #pragma omp task untied
     Solve: L[k][k] * U[k][j] = A[k][j]

  3. BDIV (Backward Division): For each block A[i][k] in column k:
     #pragma omp task untied
     Solve: L[i][k] * U[k][k] = A[i][k]

  4. #pragma omp taskwait (wait for FWD and BDIV)

  5. BMOD (Block Modification): For each block A[i][j]:
     #pragma omp task untied
     Update: A[i][j] = A[i][j] - L[i][k] * U[k][j]

  6. #pragma omp taskwait (wait for BMOD)
```

### Parallelization Strategy

Uses `#pragma omp task untied` with explicit taskwait synchronization:
- LU0 executed serially (critical path)
- FWD and BDIV tasks created in parallel
- First taskwait ensures FWD/BDIV complete before BMOD
- BMOD tasks created for all (i,j) pairs
- Second taskwait ensures BMOD complete before next iteration
- Untied tasks allow task suspension and migration

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | M=8, BS=64 | 8×8 blocks of 64×64 |
| **MEDIUM** | M=16, BS=64 | 16×16 blocks of 64×64 (default) |
| **LARGE** | M=32, BS=64 | 32×32 blocks of 64×64 |

Where M = matrix_size (number of blocks), BS = block_size (submatrix dimension)

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (M=8, BS=64)
make small

# Build medium size (M=16, BS=64) - default
make medium

# Build large size (M=32, BS=64)
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
#pragma omp parallel
#pragma omp single nowait
for (kk = 0; kk < matrix_size; kk++) {
  lu0(...);  // Serial diagonal factorization

  // Parallel FWD tasks
  for (jj = kk+1; jj < matrix_size; jj++)
    #pragma omp task untied firstprivate(kk,jj) shared(BENCH)
      fwd(...);

  // Parallel BDIV tasks
  for (ii = kk+1; ii < matrix_size; ii++)
    #pragma omp task untied firstprivate(kk,ii) shared(BENCH)
      bdiv(...);

  #pragma omp taskwait  // Wait for FWD and BDIV

  // Parallel BMOD tasks
  for (ii = kk+1; ii < matrix_size; ii++)
    for (jj = kk+1; jj < matrix_size; jj++)
      #pragma omp task untied firstprivate(kk,jj,ii) shared(BENCH)
        bmod(...);

  #pragma omp taskwait  // Wait for BMOD
}
```

**Key Features:**
- **Untied tasks**: Can be suspended and resumed on different threads
- **Fork-join pattern**: Explicit taskwait for phase synchronization
- **Implicit dependencies**: Dependencies enforced by taskwait barriers
- **Simple model**: Easier to understand than explicit depend clauses

**Advantages:**
- Simpler than explicit dependencies
- Untied allows better load balancing
- Clear phase separation with taskwait
- Good parallelism within each phase

**Trade-offs:**
- Coarser synchronization than task-dep variant
- Taskwait may limit pipelining opportunities
- Less flexible than explicit dependencies

## Comparison with Other Variants

| Aspect | sparselu-task | sparselu-task-dep |
|--------|---------------|-------------------|
| **Dependencies** | Implicit (taskwait) | Explicit (depend clauses) |
| **Synchronization** | Two taskwait per iteration | Automatic by runtime |
| **Task type** | Untied | Default (tied) |
| **Complexity** | Simpler | More complex |
| **Pipelining** | Limited (barriers) | Possible (data-flow) |
| **OpenMP version** | 3.0+ (untied) | 4.0+ (depend) |

## Four Core Kernels

### 1. LU0 - Diagonal Block Factorization
- Standard dense LU factorization of diagonal block
- Executed serially (critical path)

### 2. FWD - Forward Substitution
- Solves L * X = B for upper triangular block
- Depends on: LU0 of same block
- Parallelizable across columns

### 3. BDIV - Backward Division
- Solves X * U = B for lower triangular block
- Depends on: LU0 of same block
- Parallelizable across rows

### 4. BMOD - Block Modification (Update)
- Updates interior block: A[i][j] -= L[i][k] * U[k][j]
- Depends on: FWD and BDIV of corresponding blocks
- Highest parallelism (O(n²) tasks)

## CARTS Compatibility

- No global variables
- Matrix passed as parameter (BENCH)
- OpenMP untied tasks
- Clean block-based structure
- Fork-join task pattern
- Suitable for testing task scheduling

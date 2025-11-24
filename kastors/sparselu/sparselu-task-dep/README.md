# SparseLU-Task-Dep - Explicit Dependency Sparse LU Factorization

## Description

Block-based sparse LU factorization using OpenMP 4.0 task dependencies with explicit depend clauses. This variant expresses fine-grained data-flow dependencies between tasks, allowing the runtime to automatically schedule tasks in a pipelined fashion without explicit barriers, maximizing parallelism and enabling wavefront execution.

## Algorithm

Block-based LU factorization for sparse matrices using four kernel operations with explicit dependencies:

```
For each diagonal block k:
  1. LU0: Factor diagonal block
     #pragma omp task depend(inout: BENCH[k][k])
     A[k][k] = L[k][k] * U[k][k]

  2. FWD (Forward): For each block in row k:
     #pragma omp task depend(in: BENCH[k][k]) depend(inout: BENCH[k][j])
     Solve: L[k][k] * U[k][j] = A[k][j]

  3. BDIV (Backward Division): For each block in column k:
     #pragma omp task depend(in: BENCH[k][k]) depend(inout: BENCH[i][k])
     Solve: L[i][k] * U[k][k] = A[i][k]

  4. BMOD (Block Modification): For each interior block:
     #pragma omp task depend(in: BENCH[i][k], BENCH[k][j]) depend(inout: BENCH[i][j])
     Update: A[i][j] = A[i][j] - L[i][k] * U[k][j]

No explicit taskwait needed - runtime handles dependencies automatically!
```

### Parallelization Strategy

Uses `#pragma omp task depend` for fine-grained data-flow execution:
- Each kernel specifies input/output dependencies
- Runtime automatically schedules tasks when dependencies satisfied
- Multiple diagonal blocks can be in-flight simultaneously (pipelining)
- Wavefront execution pattern emerges naturally

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

**OpenMP task dependency approach:**

```c
#pragma omp parallel private(kk,ii,jj) shared(BENCH)
#pragma omp single
{
  for (kk = 0; kk < matrix_size; kk++) {
    // LU0: Factor diagonal block
    #pragma omp task firstprivate(kk) shared(BENCH)
    depend(inout: BENCH[kk*matrix_size+kk : submatrix_size*submatrix_size])
      lu0(BENCH[kk*matrix_size+kk], submatrix_size);

    // FWD: Forward substitution
    for (jj = kk+1; jj < matrix_size; jj++)
      if (BENCH[kk*matrix_size+jj] != NULL) {
        #pragma omp task firstprivate(kk,jj) shared(BENCH)
        depend(in: BENCH[kk*matrix_size+kk : ...])
        depend(inout: BENCH[kk*matrix_size+jj : ...])
          fwd(...);
      }

    // BDIV: Backward division
    for (ii = kk+1; ii < matrix_size; ii++)
      if (BENCH[ii*matrix_size+kk] != NULL) {
        #pragma omp task firstprivate(kk,ii) shared(BENCH)
        depend(in: BENCH[kk*matrix_size+kk : ...])
        depend(inout: BENCH[ii*matrix_size+kk : ...])
          bdiv(...);
      }

    // BMOD: Block modification
    for (ii = kk+1; ii < matrix_size; ii++)
      for (jj = kk+1; jj < matrix_size; jj++)
        if (...) {
          #pragma omp task firstprivate(kk,jj,ii) shared(BENCH)
          depend(in: BENCH[ii*matrix_size+kk : ...], BENCH[kk*matrix_size+jj : ...])
          depend(inout: BENCH[ii*matrix_size+jj : ...])
            bmod(...);
        }
  }
  #pragma omp taskwait  // Final wait for completion
}
```

**Dependency Pattern:**
```
Iteration k:
  LU0[k,k] (diagonal factorization)
    ↓ (in)              ↓ (in)
  FWD[k,j]            BDIV[i,k]
    ↓ (in)              ↓ (in)
         BMOD[i,j] (depends on both)

BMOD[i,j] in iteration k can overlap with:
  - LU0[k+1,k+1] from iteration k+1
  - FWD/BDIV from iteration k+1
  → Natural wavefront/pipelining
```

**Key Features:**
- **Data-flow execution**: Tasks scheduled by data availability
- **Array section dependencies**: `depend(... : ptr : size)` syntax
- **Automatic pipelining**: Multiple iterations overlap naturally
- **No explicit barriers**: Dependencies enforce correctness
- **Wavefront pattern**: Emerges from dependency structure

**Advantages:**
- Maximum parallelism (fine-grained dependencies)
- Natural pipelining across iterations
- No over-synchronization (vs. taskwait approach)
- Demonstrates OpenMP 4.0 capabilities

**Trade-offs:**
- More complex to implement and understand
- Runtime dependency tracking overhead
- Requires OpenMP 4.0+ support
- Debugging can be challenging

## Comparison with Other Variants

| Aspect | sparselu-task-dep | sparselu-task |
|--------|-------------------|---------------|
| **Dependencies** | Explicit (depend clauses) | Implicit (taskwait) |
| **Synchronization** | Automatic (data-flow) | Manual (barriers) |
| **Pipelining** | Yes (natural) | No (limited by barriers) |
| **Parallelism** | Maximum | Good |
| **Complexity** | Higher | Lower |
| **OpenMP version** | 4.0+ (depend) | 3.0+ (untied) |
| **Runtime overhead** | Dependency tracking | Task creation only |

## Four Core Kernels

### 1. LU0 - Diagonal Block Factorization
```c
#pragma omp task depend(inout: BENCH[k][k])
```
- Standard dense LU factorization
- Critical path (all others depend on it)
- One per iteration

### 2. FWD - Forward Substitution
```c
#pragma omp task depend(in: BENCH[k][k]) depend(inout: BENCH[k][j])
```
- Solves L * X = B for upper triangular block
- Reads: LU0[k,k]
- Writes: FWD[k,j]
- O(n) parallelism per iteration

### 3. BDIV - Backward Division
```c
#pragma omp task depend(in: BENCH[k][k]) depend(inout: BENCH[i][k])
```
- Solves X * U = B for lower triangular block
- Reads: LU0[k,k]
- Writes: BDIV[i,k]
- O(n) parallelism per iteration

### 4. BMOD - Block Modification (Update)
```c
#pragma omp task depend(in: BENCH[i][k], BENCH[k][j]) depend(inout: BENCH[i][j])
```
- Updates interior block: A[i][j] -= L[i][k] * U[k][j]
- Reads: BDIV[i,k], FWD[k,j]
- Writes: BMOD[i,j]
- O(n²) parallelism per iteration

## CARTS Compatibility

- No global variables
- Matrix passed as parameter (BENCH)
- OpenMP task dependencies (OpenMP 4.0+)
- Clean block-based structure
- Explicit data-flow dependencies
- Array section depend syntax
- Rich dependency pattern for analysis
- Demonstrates wavefront execution

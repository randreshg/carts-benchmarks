# KaStORS SparseLU - Sparse LU Factorization

## Description

Block-based sparse LU factorization using OpenMP tasks with dependencies. Decomposes a sparse matrix into lower and upper triangular factors using block operations.

## Original Source

**Repository**: [KaStORS (Karlsruhe OpenMP Tasks Suite)](https://github.com/viroulep/kastors)
**Original Name**: Barcelona OpenMP Tasks Suite (BOTS)
**License**: GNU General Public License v2.0
**Copyright**:
- Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
- Universitat Politecnica de Catalunya

**KaStORS Project**:
- **Repository**: https://github.com/viroulep/kastors
- **Purpose**: OpenMP 4.0 task dependency benchmarks
- **Features**: Task-based parallelism with explicit dependencies

## Algorithm

Block-based LU factorization for sparse matrices using four kernel operations:

```
For each diagonal block k:
  1. LU0: Factor diagonal block A[k][k] = L[k][k] * U[k][k]

  2. FWD (Forward): For each block A[k][j] in row k:
     Solve: L[k][k] * U[k][j] = A[k][j]

  3. BDIV (Backward Division): For each block A[i][k] in column k:
     Solve: L[i][k] * U[k][k] = A[i][k]

  4. BMOD (Block Modification): For each block A[i][j]:
     Update: A[i][j] = A[i][j] - L[i][k] * U[k][j]
```

### Dependency Pattern

```
    k-th iteration:
    LU0(k,k) → FWD(k,j) → BMOD(i,j)
            ↘ BDIV(i,k) ↗

    BMOD(i,j) depends on:
    - BDIV(i,k)  [for L[i][k]]
    - FWD(k,j)   [for U[k][j]]
```

## Variants

### 1. sparselu-task
- **Parallelization**: `#pragma omp task untied`
- **Dependencies**: Implicit (via taskwait)
- **Pattern**: Fork-join task creation

### 2. sparselu-task-dep
- **Parallelization**: `#pragma omp task depend(in:...) depend(out:...)`
- **Dependencies**: Explicit using OpenMP 4.0 depend clauses
- **Pattern**: Data-flow execution with fine-grained dependencies

### 3. sparselu-task-carts / sparselu-task-dep-carts
- **Description**: CARTS-adapted versions
- **Modifications**: Adjusted for CARTS compiler compatibility

## Four Core Kernels

### 1. LU0 - Diagonal Block Factorization
```c
void lu0(float *diag, int N)
```
- Standard dense LU factorization of diagonal block
- No dependencies (first operation in each iteration)

### 2. FWD - Forward Substitution
```c
void fwd(float *diag, float *col, int N)
```
- Solves L * X = B for upper triangular block
- Depends on: LU0 of same block

### 3. BDIV - Backward Division
```c
void bdiv(float *diag, float *row, int N)
```
- Solves X * U = B for lower triangular block
- Depends on: LU0 of same block

### 4. BMOD - Block Modification (Update)
```c
void bmod(float *row, float *col, float *inner, int N)
```
- Updates interior block: A[i][j] -= L[i][k] * U[k][j]
- Depends on: FWD and BDIV of corresponding blocks

## Build

```bash
cd kastors/sparselu/

# Task version (untied tasks)
cd sparselu-task/
make

# Task-dependency version (OpenMP 4.0)
cd sparselu-task-dep/
make
```

## Usage

```bash
# SparseLU task-dep version
./sparselu-task-dep <matrix_size> <block_size>

# Example: 8×8 blocks of size 64
./sparselu-task-dep 8 64
```

## Mathematical Background

**LU Factorization**: Decomposes matrix A = L × U
- **L**: Lower triangular matrix (with 1s on diagonal)
- **U**: Upper triangular matrix

For sparse matrices:
- Only non-zero blocks are stored
- Block structure exploits sparsity pattern
- Reduces memory and computation

**Block Algorithm**:
- Divides matrix into blocks
- Each block operation is a dense matrix operation
- Dependencies between blocks allow parallelization

## Use in Computing

Sparse LU factorization is fundamental in:
- **Linear Solvers**: Solving Ax = b for sparse systems
- **Structural Analysis**: Finite element methods
- **Circuit Simulation**: SPICE and electronic design
- **Computational Chemistry**: Molecular dynamics
- **Graph Algorithms**: Network flow problems

## CARTS Compatibility

- No global variables (matrix passed as parameter)
- Clean block-based structure
- OpenMP task dependencies (sparselu-task-dep)
- OpenMP tasks with untied (sparselu-task)
- Rich dependency pattern for testing

## Key Features

- **Block-based**: Operates on submatrix blocks
- **Sparse structure**: Only processes non-NULL blocks
- **Task parallelism**: Natural task decomposition
- **Data-flow execution**: Dependencies drive scheduling
- **Four distinct kernels**: Different computational patterns

## CARTS Testing Focus

### Memory Access Patterns
- **Block-strided**: Access patterns within blocks
- **Irregular**: Sparse structure creates non-uniform access
- **Read-modify-write**: BMOD kernel updates in place

### Dependencies
- **Producer-consumer**: FWD/BDIV produce, BMOD consumes
- **Wavefront**: Diagonal blocks create wavefront pattern
- **Fine-grained**: Block-level dependencies

### Parallelization Opportunities
- **Task parallelism**: Independent blocks in parallel
- **Pipeline**: Multiple diagonal blocks in flight
- **Load balancing**: Irregular sparsity affects work distribution

## Performance Characteristics

- **Compute**: O(n³) for dense, reduced by sparsity
- **Memory**: Only non-zero blocks stored
- **Parallelism**: Limited by diagonal blocks (critical path)
- **Cache**: Block size affects cache efficiency

## Comparison with PLASMA/LAPACK

| Aspect | KaStORS SparseLU | PLASMA/LAPACK |
|--------|------------------|---------------|
| **Matrix Type** | Sparse (blocked) | Dense |
| **Parallelism** | Task-based | Thread-based |
| **Dependencies** | Explicit (OpenMP 4.0) | Implicit (barriers) |
| **Block Operations** | 4 kernels (LU0, FWD, BDIV, BMOD) | Similar DAG structure |

## References

- **KaStORS Repository**: https://github.com/viroulep/kastors
- **BOTS (Original)**: https://github.com/bsc-pm/bots
- **IWOMP 2014 Paper**: Virouleau et al., "Evaluation of OpenMP dependent tasks with the KASTORS benchmark suite"
- **Sparse LU**: Classic algorithm for sparse linear systems
- **OpenMP 4.0**: Task dependencies introduced in OpenMP 4.0 (2013)

## Citation

### KaStORS
```
Virouleau, Philippe, et al.
"Evaluation of OpenMP dependent tasks with the KASTORS benchmark suite."
International Workshop on OpenMP (IWOMP), 2014.
```

### Original BOTS
```
Duran, Alejandro, et al.
"Barcelona OpenMP Tasks Suite: A set of benchmarks targeting the exploitation of task parallelism in OpenMP."
International Conference on Parallel Processing (ICPP), 2009.
```

### Sparse Direct Solvers
```
Davis, Timothy A.
"Direct Methods for Sparse Linear Systems."
SIAM, 2006.
```

### Block LU Factorization
```
Buttari, Alfredo, et al.
"A class of parallel tiled linear algebra algorithms for multicore architectures."
Parallel Computing, 2009.
```

# KaStORS Strassen - Fast Matrix Multiplication

## Description

Strassen's algorithm for fast matrix multiplication using divide-and-conquer with OpenMP task parallelism. Reduces arithmetic complexity from O(n³) to O(n^2.807).

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

Strassen's algorithm divides n×n matrices into 2×2 blocks and computes 7 products instead of 8:

### Traditional Matrix Multiplication (n×n)
```
C = A × B requires n³ multiplications

For blocks:
C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

Total: 8 multiplications, 4 additions
```

### Strassen's Algorithm
```
Instead, compute 7 products:
M1 = (A11 + A22) * (B11 + B22)
M2 = (A21 + A22) * B11
M3 = A11 * (B12 - B22)
M4 = A22 * (B21 - B11)
M5 = (A11 + A12) * B22
M6 = (A21 - A11) * (B11 + B12)
M7 = (A12 - A22) * (B21 + B22)

Then combine:
C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

Total: 7 multiplications, 18 additions
Complexity: O(n^log₂(7)) ≈ O(n^2.807)
```

### Recursive Strategy
```
1. Base case: Use standard multiplication for small matrices
2. Recursive case:
   - Split matrices into quadrants
   - Compute 7 Strassen products recursively
   - Combine results
```

## Variants

### 1. strassen-task
- **Parallelization**: `#pragma omp task untied`
- **Dependencies**: Implicit (via taskwait)
- **Pattern**: Recursive task creation

### 2. strassen-task-dep
- **Parallelization**: `#pragma omp task depend(in:...) depend(out:...)`
- **Dependencies**: Explicit using OpenMP 4.0 depend clauses
- **Pattern**: Data-flow execution with explicit dependencies

### 3. main-strassen
- **Description**: Driver program with problem setup
- **Includes**: Matrix allocation, initialization, validation

## Build

```bash
cd kastors/strassen/

# Task version
cd strassen-task/
make

# Task-dependency version
cd strassen-task-dep/
make
```

## Usage

```bash
# Strassen task-dep version
./strassen-task-dep <matrix_size>

# Example: 1024×1024 matrices
./strassen-task-dep 1024
```

**Note**: Matrix size should typically be a power of 2 for optimal performance.

## Mathematical Background

**Discovered**: Volker Strassen (1969)

**Key Insight**: Matrix multiplication can be done with fewer multiplications than the obvious n³ algorithm.

**Why 7 products?**:
- Traditional block multiplication: 8 products
- Strassen found a clever combination that needs only 7
- Multiplications are more expensive than additions
- Asymptotic improvement: O(n^2.807) vs O(n³)

**Practical Considerations**:
- Crossover point: ~64-128 (switch to standard below this)
- Numerical stability: Slightly worse than standard
- Cache behavior: Recursive structure can be cache-friendly

## Use in Computing

Strassen's algorithm influenced:
- **Fast Matrix Multiplication**: Started a field of research
- **Coppersmith-Winograd**: Further improvements to O(n^2.376)
- **Practical Libraries**: Used in some contexts (e.g., large matrices)
- **Theoretical CS**: Showed naive algorithms aren't always optimal

**Modern Alternatives**:
- **Winograd Variant**: O(n^2.373)
- **Le Gall's Algorithm** (2014): O(n^2.3728639)
- **Practical**: Often standard GEMM is faster due to constants and optimizations

## CARTS Compatibility

- No global variables (matrices passed as parameters)
- Clean recursive structure
- OpenMP task parallelism
- OpenMP task dependencies (strassen-task-dep)
- Divide-and-conquer pattern

## Key Features

- **Recursive decomposition**: Natural divide-and-conquer
- **Task parallelism**: 7 independent products can run in parallel
- **Arithmetic savings**: Fewer multiplications (more additions)
- **Asymptotic improvement**: O(n^2.807) vs O(n³)
- **Base case**: Switches to standard multiplication for small blocks

## CARTS Testing Focus

### Memory Access Patterns
- **Recursive**: Stack-based recursion with matrix views
- **Block-strided**: Quadrant accesses
- **Temporary matrices**: 7 intermediate products

### Dependencies
- **7-way parallelism**: M1-M7 can execute in parallel
- **Combination phase**: C11-C22 depend on M products
- **Recursive**: Parent tasks depend on child tasks

### Parallelization Opportunities
- **Horizontal**: 7 products in each level
- **Vertical**: Recursive levels can overlap
- **Load balancing**: Work distribution across tasks

## Performance Characteristics

- **Compute**: O(n^2.807) arithmetic operations
- **Memory**: Temporary matrices for M1-M7
- **Parallelism**: Exponentially increasing tasks (7^depth)
- **Constants**: High constants make it slower for small n
- **Crossover**: Usually around n=128-256

## Comparison with Standard GEMM

| Aspect | Strassen | Standard GEMM |
|--------|----------|---------------|
| **Complexity** | O(n^2.807) | O(n³) |
| **Multiplications** | Fewer | More |
| **Additions** | More | Fewer |
| **Numerical Stability** | Slightly worse | Better |
| **Cache Behavior** | Good (recursive) | Excellent (optimized) |
| **Practical Speed** | Good for large n | Good for small n |

## Historical Context

**1969**: Strassen publishes algorithm
- First improvement to matrix multiplication since antiquity
- Showed O(n³) is not optimal
- Opened field of fast matrix multiplication

**Impact**:
- Theoretical: Inspired research in algorithm design
- Practical: Used in some libraries for very large matrices
- Educational: Classic example of divide-and-conquer

## References

- **KaStORS Repository**: https://github.com/viroulep/kastors
- **BOTS (Original)**: https://github.com/bsc-pm/bots
- **Strassen's Paper** (1969): "Gaussian Elimination is not Optimal"
- **Modern Improvements**: Le Gall (2014), Williams et al.

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

### Strassen's Algorithm
```
Strassen, Volker.
"Gaussian elimination is not optimal."
Numerische Mathematik 13.4 (1969): 354-356.
```

### Modern Fast Matrix Multiplication
```
Le Gall, François.
"Powers of tensors and fast matrix multiplication."
International Symposium on Symbolic and Algebraic Computation (ISSAC), 2014.
```

### Fast Matrix Multiplication Survey
```
Ballard, Grey, et al.
"Communication-optimal parallel algorithm for strassen's matrix multiplication."
ACM Symposium on Parallelism in Algorithms and Architectures (SPAA), 2012.
```

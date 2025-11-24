# Strassen-Task-Dep - Explicit Dependency Strassen Algorithm

## Description

Strassen's fast matrix multiplication algorithm using OpenMP 4.0 task dependencies with explicit depend clauses. This variant implements the recursive divide-and-conquer Strassen algorithm with fine-grained data-flow dependencies, allowing the runtime to automatically schedule tasks based on data availability without explicit barriers. Reduces arithmetic complexity from O(n³) to O(n^2.807) while maximizing parallelism through dependency-driven execution.

## Algorithm

Strassen's algorithm divides n×n matrices into 2×2 blocks and computes 7 products instead of 8:

```
Traditional block multiplication: 8 products
Strassen: 7 products with clever combinations + explicit dependencies

Step 1: Compute 8 temporary matrices (S1-S8) with dependencies:
  #pragma omp task depend(in: A21, A22) depend(out: S1)
    S1 = A21 + A22
  #pragma omp task depend(in: S1, A) depend(out: S2)
    S2 = S1 - A
  #pragma omp task depend(in: A12, S2) depend(out: S4)
    S4 = A12 - S2
  ... (dependency chain ensures correct order)

Step 2: Compute 7 products recursively with dependencies:
  #pragma omp task depend(in: A, B) depend(out: M2)
    M2 = A * B
  #pragma omp task untied depend(in: S1, S5) depend(out: M5)
    M5 = S1 * S5
  ... (tasks execute when inputs available)

Step 3: Combine to form result with dependencies:
  #pragma omp task depend(inout: C) depend(in: M2)
    C += M2
  #pragma omp task depend(inout: C12) depend(in: M5, T1sMULT, M2)
    C12 += M5 + T1sMULT + M2
  ... (final combination respects dependencies)

Base case: MatrixSize <= cutoff_size → standard multiplication
No explicit taskwait needed - dependencies handle everything!
```

### Parallelization Strategy

Uses `#pragma omp task depend` for automatic data-flow execution:
- Each computation specifies input (in) and output (out/inout) dependencies
- Runtime schedules tasks when all inputs are available
- No explicit barriers - dependencies ensure correctness
- Natural pipelining across recursive levels
- Untied tasks for recursive multiplications enable migration

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

**OpenMP task dependency approach:**

```c
if (Depth < cutoff_depth) {
  // Phase 1: Compute S1-S8 with explicit dependencies
  #pragma omp task depend(in: A21, A22) depend(out: S1) private(Row,Column)
    for (...) S1[...] = A21[...] + A22[...];

  #pragma omp task depend(in: S1, A) depend(out: S2) private(Row,Column)
    for (...) S2[...] = S1[...] - A[...];

  #pragma omp task depend(in: A12, S2) depend(out: S4) private(Row,Column)
    for (...) S4[...] = A12[...] - S2[...];

  #pragma omp task depend(in: B12, B) depend(out: S5) private(Row,Column)
    for (...) S5[...] = B12[...] - B[...];

  #pragma omp task depend(in: B22, S5) depend(out: S6) private(Row,Column)
    for (...) S6[...] = B22[...] - S5[...];

  #pragma omp task depend(in: S6, B21) depend(out: S8) private(Row,Column)
    for (...) S8[...] = S6[...] - B21[...];

  #pragma omp task depend(in: A, A21) depend(out: S3) private(Row,Column)
    for (...) S3[...] = A[...] - A21[...];

  #pragma omp task depend(in: B22, B12) depend(out: S7) private(Row,Column)
    for (...) S7[...] = B22[...] - B12[...];

  // Phase 2: 7 recursive multiplications with dependencies
  #pragma omp task depend(in: A, B) depend(out: M2)
    OptimizedStrassenMultiply_par(M2, A, B, ...);

  #pragma omp task untied depend(in: S1, S5) depend(out: M5)
    OptimizedStrassenMultiply_par(M5, S1, S5, ...);

  #pragma omp task untied depend(in: S2, S6) depend(out: T1sMULT)
    OptimizedStrassenMultiply_par(T1sMULT, S2, S6, ...);

  #pragma omp task untied depend(in: S3, S7) depend(out: C22)
    OptimizedStrassenMultiply_par(C22, S3, S7, ...);

  #pragma omp task untied depend(in: A12, B21) depend(out: C)
    OptimizedStrassenMultiply_par(C, A12, B21, ...);

  #pragma omp task untied depend(in: S4, B22) depend(out: C12)
    OptimizedStrassenMultiply_par(C12, S4, B22, ...);

  #pragma omp task untied depend(in: A22, S8) depend(out: C21)
    OptimizedStrassenMultiply_par(C21, A22, S8, ...);

  // Phase 3: Combine results with dependencies
  #pragma omp task depend(inout: C) depend(in: M2) private(Row,Column)
    for (...) C[...] += M2[...];

  #pragma omp task depend(inout: C12) depend(in: M5, T1sMULT, M2) private(Row,Column)
    for (...) C12[...] += M5[...] + T1sMULT[...] + M2[...];

  #pragma omp task depend(inout: C21) depend(in: C22, T1sMULT, M2) private(Row,Column)
    for (...) C21[...] = -C21[...] + C22[...] + T1sMULT[...] + M2[...];

  #pragma omp task depend(inout: C22) depend(in: M5, T1sMULT, M2) private(Row,Column)
    for (...) C22[...] += M5[...] + T1sMULT[...] + M2[...];

  #pragma omp taskwait  // Only at end for final synchronization
} else {
  // Sequential computation (depth limit reached)
}
```

**Dependency Pattern:**

```
Level 1 (S matrices):
  A21,A22 → S1 → S2 → S4
  B12,B → S5 → S6 → S8
  A,A21 → S3
  B22,B12 → S7

Level 2 (M matrices - 7 parallel branches):
  A,B → M2 ─┐
  S1,S5 → M5 ─┤
  S2,S6 → T1sMULT ─┤
  S3,S7 → C22 ─┤  → Level 3 (Combinations)
  A12,B21 → C ─┤
  S4,B22 → C12 ─┤
  A22,S8 → C21 ─┘

Level 3 (C combinations):
  M2 → C
  M5,T1sMULT,M2 → C12
  C22,T1sMULT,M2 → C21
  M5,T1sMULT,M2 → C22
```

**Key Features:**
- **Data-flow execution**: Tasks run when inputs ready
- **No explicit barriers**: Dependencies handle synchronization
- **Natural pipelining**: Multiple levels can overlap
- **Automatic scheduling**: Runtime finds parallelism
- **Fine-grained**: Expresses exact read/write dependencies

**Advantages:**
- Maximum parallelism (no over-synchronization)
- Natural pipelining across recursive levels
- Elegant expression of algorithm structure
- Runtime optimizes scheduling automatically
- Demonstrates OpenMP 4.0 capabilities

**Trade-offs:**
- More complex to implement and debug
- Higher runtime dependency tracking overhead
- Requires OpenMP 4.0+ support
- Dependency specification can be error-prone

## Comparison with Other Variants

| Aspect | strassen-task-dep | strassen-task |
|--------|-------------------|---------------|
| **Dependencies** | Explicit (depend clauses) | Implicit (taskwait) |
| **Synchronization** | Automatic (data-flow) | Manual (barriers) |
| **Barriers** | One at end | Three per level |
| **Pipelining** | Yes (natural) | Limited |
| **Parallelism** | Maximum | Good |
| **Complexity** | Higher (depend specs) | Lower (taskwait) |
| **Runtime overhead** | Dependency tracking | Task creation only |
| **OpenMP version** | 4.0+ (depend) | 3.0+ (untied) |

## Strassen Algorithm Mathematics

**Complexity:**
- Traditional: O(n³) arithmetic operations
- Strassen: O(n^log₂(7)) ≈ O(n^2.807)
- Asymptotic improvement: ~31% fewer operations for large n

**Why 7 products?**
- Discovered by Volker Strassen (1969)
- Exploits algebraic identities:
  ```
  Instead of 8 multiplications for block matrix multiply,
  use 7 multiplications with additional additions
  ```
- First algorithm to beat O(n³) for matrix multiplication

**Recursion:**
- Each level: 7 recursive calls (depth d → 7^d subproblems)
- Base case: Switch to standard multiplication at cutoff_size
- Depth: log₂(n/cutoff_size)

## Performance Characteristics

- **Compute**: O(n^2.807) vs O(n³)
- **Memory**: Higher (temporary matrices S1-S8, M2, M5, T1sMULT)
- **Parallelism**: 7^depth potential tasks (exponential)
- **Dependencies**: Runtime tracks ~20+ dependencies per level
- **Crossover**: Typically faster for n>128
- **Cache**: Recursive structure can be cache-friendly

## Dependency Analysis Value

This benchmark is particularly valuable for CARTS because:
1. **Rich dependency graph**: Complex data-flow patterns
2. **Recursive structure**: Dependencies span multiple levels
3. **Mixed patterns**: Sequential chains + parallel branches
4. **Real algorithm**: Not synthetic, actual optimization technique
5. **Scalable**: Complexity grows with matrix size and depth

## CARTS Compatibility

- No global variables
- Matrices passed as parameters
- OpenMP task dependencies (OpenMP 4.0+)
- Clean recursive structure
- Explicit data-flow dependencies
- Divide-and-conquer pattern
- Rich dependency patterns for analysis
- Demonstrates advanced task features
- Natural pipelining opportunities

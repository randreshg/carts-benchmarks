# Jacobi-For - Parallel-For Jacobi Iteration

## Description

Iterative Jacobi solver for the 2D Poisson equation using OpenMP parallel-for directives. This variant uses simple data-parallel loops for row-wise parallelization across the grid, making it suitable for straightforward parallelization without task dependencies.

## Algorithm

Solves the Poisson equation ∇²u = f on a 2D grid using Jacobi iteration with a 5-point stencil:

```
For each iteration:
  1. Save current estimate: u_old = u
  2. Update interior points:
     u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                       u_old[i][j-1] + u_old[i][j+1] +
                       f[i][j] * dx * dy)
  3. Keep boundary conditions: u[boundary] = f[boundary]
  4. Repeat until convergence
```

### Parallelization Strategy

Uses `#pragma omp parallel for` to parallelize across grid rows:
- Two parallel loops per iteration: one for copying u to unew, one for computing new values
- Each thread processes multiple rows independently
- Implicit barrier synchronization between loops

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | N=512 | Small problem size |
| **MEDIUM** | N=1024 | Medium problem size (default) |
| **LARGE** | N=2048 | Large problem size |

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (N=512)
make small

# Build medium size (N=1024) - default
make medium

# Build large size (N=2048)
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

**OpenMP parallel-for approach:**
- Two `#pragma omp parallel for private(j)` loops per iteration
- First loop: Copy unew to u (save current estimate)
- Second loop: Compute new values using 5-point stencil
- Implicit barrier synchronization ensures all threads complete before next iteration

**Advantages:**
- Simple to implement and understand
- Good load balancing for regular grids
- Low task creation overhead

**Trade-offs:**
- Barrier synchronization required between phases
- Less flexible than task-based approaches
- Better suited for shared-memory multi-core systems

## Comparison with Other Variants

| Aspect | jacobi-for | jacobi-task-dep |
|--------|-----------|-----------------|
| **Parallelization** | `#pragma omp for` | `#pragma omp task depend` |
| **Dependencies** | Implicit barriers | Explicit dependencies |
| **Granularity** | Row-level | Row-level with dependencies |
| **Synchronization** | Barriers | Task dependencies |

## CARTS Compatibility

- No global variables
- Clean parameter passing via function arguments
- OpenMP parallel-for directives
- VLA casting for clean array indexing
- Standard 5-point stencil pattern

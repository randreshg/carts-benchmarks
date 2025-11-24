# Poisson-For - Parallel-For Poisson Solver

## Description

Iterative Poisson equation solver using Jacobi method with OpenMP parallel-for directives. This variant solves the Poisson equation ∇²u = f with specific boundary conditions (u(x,y) = sin(πxy)) and includes validation against the exact solution. Uses simple data-parallel loops for straightforward parallelization.

## Algorithm

Solves the Poisson equation ∇²u = f on the unit square [0,1] × [0,1]:

```
Exact solution: u(x,y) = sin(π * x * y)
Right-hand side: f(x,y) = -π² * (x² + y²) * sin(π * x * y)

For each iteration:
  1. Save current estimate: u_old = u
  2. Update interior points using 5-point stencil:
     u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                       u_old[i][j-1] + u_old[i][j+1] +
                       f[i][j] * dx * dy)
  3. Boundary: u[boundary] = f[boundary] (Dirichlet conditions)
  4. Repeat until convergence or max iterations
```

### Parallelization Strategy

Uses `#pragma omp parallel for` for row-wise parallelization:
- Initialization parallelized with `collapse(2)` for 2D blocking
- Sweep function uses two parallel loops per iteration
- Simple barrier synchronization between phases

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | N=512, block=128 | Small problem size |
| **MEDIUM** | N=1024, block=128 | Medium problem size (default) |
| **LARGE** | N=2048, block=128 | Large problem size |

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
- Initialization uses `#pragma omp parallel for collapse(2)` for 2D block parallelization
- Sweep function uses two `#pragma omp parallel for private(j)` loops per iteration:
  - First loop: Copy unew to u (save current estimate)
  - Second loop: Compute new values using 5-point stencil
- Implicit barrier synchronization ensures consistency

**Key Features:**
- Block-based initialization (block_size parameter)
- Row-wise parallelization in main iteration
- Clean separation of boundary and interior point handling
- Validation against analytical solution

## Comparison with Other Variants

| Aspect | poisson-for | poisson-task |
|--------|------------|--------------|
| **Parallelization** | `#pragma omp for` | `#pragma omp task` |
| **Dependencies** | Implicit barriers | Explicit task dependencies |
| **Initialization** | Parallel for collapse | Task-based |
| **Complexity** | Simple | More complex |
| **Best for** | Multi-core SMP | Task scheduling study |

## Mathematical Background

**Poisson Equation:** ∇²u = f

In 2D with finite differences (5-point stencil):
```
(u[i-1][j] - 2*u[i][j] + u[i+1][j])/dx² +
(u[i][j-1] - 2*u[i][j] + u[i][j+1])/dy² = f[i][j]
```

Rearranging for u[i][j] (assuming dx = dy):
```
u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] + f[i][j]*h²)
```

## CARTS Compatibility

- No global variables
- Clean parameter passing via function arguments
- OpenMP parallel-for directives
- VLA casting for clean array indexing
- Block-based initialization
- Analytical solution validation

# Poisson-Task - Task-Based Poisson Solver

## Description

Iterative Poisson equation solver using Jacobi method with OpenMP task parallelism. This variant solves the Poisson equation ∇²u = f with specific boundary conditions (u(x,y) = sin(πxy)) using task-based parallelization for both initialization and computation phases, providing fine-grained parallelism without explicit dependencies.

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

Uses `#pragma omp task` for block-based parallelization:
- Initialization creates tasks for each block of the grid
- RHS computation also uses task-based blocking
- Tasks synchronized with implicit barriers (no explicit dependencies)
- Block size controls task granularity

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

**OpenMP task approach:**
- `#pragma omp parallel` / `#pragma omp master` pattern
- Block-based task creation for initialization and RHS:
  ```c
  for (j = 0; j < ny; j += block_size)
    for (i = 0; i < nx; i += block_size)
      #pragma omp task firstprivate(i,j) private(ii,jj)
  ```
- Sweep function (presumably similar to jacobi-task-dep) uses task-based iteration
- Implicit synchronization via taskwait

**Key Features:**
- Block-level task granularity (controlled by block_size parameter)
- Task creation in master thread
- Firstprivate clause for loop indices
- Clean separation of task creation and computation

**Advantages:**
- Fine-grained task parallelism
- Flexible task scheduling by runtime
- Good for load balancing with blocks
- Natural fit for heterogeneous systems

**Trade-offs:**
- Higher overhead than parallel-for
- Block size affects performance significantly
- More complex than simple parallel loops

## Comparison with Other Variants

| Aspect | poisson-task | poisson-for |
|--------|-------------|------------|
| **Parallelization** | `#pragma omp task` | `#pragma omp for` |
| **Granularity** | Block-level tasks | Row-level threads |
| **Synchronization** | Taskwait (implicit) | Barriers (implicit) |
| **Overhead** | Higher (task creation) | Lower (thread reuse) |
| **Flexibility** | High (dynamic scheduling) | Medium (static/dynamic) |
| **Block parameter** | Required (controls tasks) | Optional (for init) |

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
- OpenMP task parallelism
- VLA casting for clean array indexing
- Block-based task decomposition
- Analytical solution validation

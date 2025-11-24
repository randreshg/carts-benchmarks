# Jacobi-Task-Dep - Task Dependency Jacobi Iteration

## Description

Iterative Jacobi solver for the 2D Poisson equation using OpenMP task dependencies (OpenMP 4.0+). This variant expresses explicit data dependencies between tasks for fine-grained parallel control, enabling more sophisticated scheduling and potential for pipelining across iterations.

## Algorithm

Solves the Poisson equation ∇²u = f on a 2D grid using Jacobi iteration with a 5-point stencil:

```
For each iteration:
  1. Save current estimate: u_old = u (via task dependencies)
  2. Update interior points with stencil:
     u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                       u_old[i][j-1] + u_old[i][j+1] +
                       f[i][j] * dx * dy)
  3. Keep boundary conditions: u[boundary] = f[boundary]
  4. Repeat with explicit dependencies
```

### Parallelization Strategy

Uses `#pragma omp task depend` to express row-level dependencies:
- Copy phase: `depend(in: unew[i]) depend(out: u[i])`
- Compute phase: `depend(in: f[i], u[i-1], u[i], u[i+1]) depend(out: unew[i])`
- Runtime automatically schedules tasks respecting dependencies
- Potential for wave-front execution across the grid

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

**OpenMP task dependency approach:**
- Creates tasks for each row with explicit dependencies
- Copy tasks: Read from unew[i], write to u[i]
- Compute tasks: Read from u[i-1], u[i], u[i+1], f[i], write to unew[i]
- OpenMP runtime schedules tasks respecting dependencies

**Dependency Pattern:**
```
Iteration k:
  Copy[i]: unew[i] → u[i]
  Compute[i]: u[i-1], u[i], u[i+1] → unew[i]

Dependencies ensure:
  - Compute[i] waits for Copy[i-1], Copy[i], Copy[i+1]
  - Next iteration's Copy[i] waits for current Compute[i]
```

**Advantages:**
- Fine-grained dependency tracking
- Potential for pipelining across iterations
- More flexible scheduling than barriers
- Better suited for heterogeneous systems

**Trade-offs:**
- Higher task creation overhead
- More complex to implement and debug
- Runtime dependency tracking overhead

## Comparison with Other Variants

| Aspect | jacobi-task-dep | jacobi-for |
|--------|----------------|-----------|
| **Parallelization** | `#pragma omp task depend` | `#pragma omp for` |
| **Dependencies** | Explicit (in/out clauses) | Implicit (barriers) |
| **Scheduling** | Dynamic (runtime) | Static/dynamic |
| **Overhead** | Higher (task creation) | Lower (thread reuse) |
| **Flexibility** | High (pipelining possible) | Low (barrier-bound) |

## CARTS Compatibility

- No global variables
- Clean parameter passing via function arguments
- OpenMP task dependencies (OpenMP 4.0+)
- VLA casting for clean array indexing
- Explicit dependency specification
- Row-level dependency granularity

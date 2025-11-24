# Main-Strassen - Strassen Driver Program

## Description

Simple driver program for Strassen matrix multiplication benchmarks. This provides the main entry point and parameter configuration for the Strassen algorithm variants, setting up matrix sizes, cutoff parameters, and test configurations. It serves as the infrastructure for the actual Strassen implementations in the sibling directories.

## Purpose

This is a minimal driver that:
- Defines default parameters (matrix size, cutoff size, cutoff depth)
- Provides structure for benchmark harness integration
- Sets up user parameters for Strassen variants
- Serves as build infrastructure for standalone testing

## Default Configuration

```c
matrix_size = 128      // 128×128 matrices
cutoff_size = 32       // Switch to standard mult at 32×32
cutoff_depth = 2       // Maximum recursion depth for tasks
niter = 1              // Number of iterations
check = 0              // Validation disabled by default
```

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

## Parameters

### Matrix Size (MSIZE)
- Controls the dimension of input matrices (n×n)
- Should typically be a power of 2 for optimal performance
- Affects total computation: O(n^2.807)

### Cutoff Size (CUTOFF_SIZE)
- Size at which to switch from Strassen to standard multiplication
- Typical values: 32-128
- Smaller values: More recursion, higher overhead
- Larger values: Less benefit from Strassen, but lower overhead

### Cutoff Depth (CUTOFF_DEPTH)
- Maximum recursion depth for task creation
- Controls parallelism granularity
- Depth 1: Only top-level creates tasks (7 parallel tasks)
- Depth 2: Top two levels create tasks (7 + 49 = 56 parallel tasks)
- Higher depth: Exponentially more tasks (7^depth)

## Relationship to Other Variants

This driver program works with:
1. **strassen-task**: Untied tasks without explicit dependencies
2. **strassen-task-dep**: Tasks with explicit data-flow dependencies

The actual Strassen algorithm implementation is in those directories, while this provides:
- Parameter definitions
- Main entry point
- Integration with benchmark framework

## Strassen Algorithm Overview

Strassen's algorithm reduces matrix multiplication from O(n³) to O(n^2.807):

```
Traditional: C = A × B requires 8 block multiplications
Strassen: Computes 7 products (M1-M7) instead:

M1 = (A11 + A22) * (B11 + B22)
M2 = (A21 + A22) * B11
M3 = A11 * (B12 - B22)
M4 = A22 * (B21 - B11)
M5 = (A11 + A12) * B22
M6 = (A21 - A11) * (B11 + B12)
M7 = (A12 - A22) * (B21 + B22)

Then combines them to form C11, C12, C21, C22
```

## Performance Considerations

**Cutoff Size Trade-offs:**
- Too small: High task creation overhead, poor cache behavior
- Too large: Less benefit from Strassen's algorithm
- Optimal: Typically 32-128 depending on architecture

**Cutoff Depth Trade-offs:**
- Low depth: Less parallelism, better cache locality
- High depth: More parallelism, higher overhead, more memory
- Typical: 1-3 for good balance

**Matrix Size:**
- Power of 2: Clean recursion, no padding needed
- Non-power of 2: May require padding or special handling
- Large sizes: Greater benefit from Strassen's complexity improvement

## CARTS Compatibility

- Simple parameter structure
- Clean interface to algorithm implementations
- No global state
- Configurable problem sizes
- Integration with benchmark framework

## Usage Pattern

```c
struct user_parameters p;
p.matrix_size = 128;     // N×N matrices
p.cutoff_size = 32;      // Base case threshold
p.cutoff_depth = 2;      // Task creation depth
p.niter = 1;             // Iterations
p.check = 0;             // Validation off

// Call strassen-task or strassen-task-dep implementations
// with these parameters
```

## Notes

- This is primarily infrastructure, not the algorithm itself
- See `strassen-task/` or `strassen-task-dep/` for actual implementations
- Matrix sizes should be powers of 2 for best performance
- Cutoff parameters significantly affect performance
- Balance between parallelism (depth) and overhead is critical

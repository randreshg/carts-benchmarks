# Merge Tasks - Task-Parallel Merge Sort with OpenMP

## Description

A parallel implementation of merge sort using OpenMP task-based parallelism. This benchmark demonstrates divide-and-conquer parallelism where recursive sorting of subarrays is distributed across multiple threads as independent tasks. Note: This benchmark is from Andrew Vaillancourt's task-parallelism-omp suite.

## Algorithm

**Task-Parallel Merge Sort** uses OpenMP tasks to exploit parallelism in the divide-and-conquer pattern:
1. **Divide**: Recursively split the array into two halves, creating OpenMP tasks for each half
2. **Parallel Execution**: Each task can run on different threads, sorting subarrays independently
3. **Threshold-based Serial**: When subarray size drops below threshold, switch to serial merge sort with insertion sort optimization
4. **Synchronize**: Use `#pragma omp taskwait` to ensure both halves are sorted before merging
5. **Merge**: Combine the sorted halves sequentially (merging is inherently sequential)

The algorithm adaptively switches between parallel task creation and serial execution based on the threshold parameter, balancing parallelism overhead against available work.

## Problem Sizes

Note: These benchmarks use runtime parameters rather than compile-time sizes.

| Size | Array Size | Threshold | Threads | Description |
|------|------------|-----------|---------|-------------|
| **SMALL** | 10,000 | 100 | 4 | Small problem size for testing |
| **MEDIUM** | 100,000 | 500 | 8 | Medium problem size |
| **LARGE** | 1,000,000 | 1000 | 16 | Large problem size |
| **EXTRA LARGE** | 10,000,000 | 2000 | 32 | Very large problem for scalability testing |

**Parameters**:
- **Array Size**: Number of elements to sort
- **Threshold**: Subarray size below which serial sorting is used (controls task granularity)
- **Threads**: Number of OpenMP threads to use for parallel execution

## Building and Running

### Build with CARTS pipeline

```bash
# Build all pipeline stages
make all

# Generate sequential MLIR
make seq

# Collect runtime metadata
make metadata

# Generate parallel MLIR
make parallel
```

### Run with different sizes

After building, the MLIR can be executed with CARTS:

```bash
# Small size
carts run build/merge-tasks.mlir -- 10000 100 4

# Medium size
carts run build/merge-tasks.mlir -- 100000 500 8

# Large size
carts run build/merge-tasks.mlir -- 1000000 1000 16

# Extra large size
carts run build/merge-tasks.mlir -- 10000000 2000 32
```

### Clean build artifacts

```bash
make clean
```

## Parallelization Strategy

This benchmark uses **OpenMP task-based parallelism** with several key features:

### Task Creation Pattern
```c
#pragma omp parallel
{
  #pragma omp single nowait
  {
    #pragma omp task
    mergesort_parallel_omp(left_half, ...);

    #pragma omp task
    mergesort_parallel_omp(right_half, ...);

    #pragma omp taskwait
    merge(array, ...);
  }
}
```

### Key Parallelization Features
- **Fine-grained Tasks**: Creates tasks for each recursive split, allowing dynamic load balancing
- **Nested Parallelism**: Enables tasks to spawn additional tasks (`omp_set_nested(1)`)
- **Thread Division**: Divides available threads among subtasks (threads/2 for each half)
- **Adaptive Granularity**: Uses threshold to prevent excessive task creation overhead
- **Task Synchronization**: `taskwait` ensures both subtasks complete before merging

### Performance Considerations
- **Threshold Tuning**: Lower threshold creates more tasks but increases overhead
- **Thread Scaling**: Divides threads logarithmically across recursion levels
- **Memory Locality**: Each task works on contiguous memory regions
- **Load Balancing**: OpenMP runtime dynamically assigns tasks to idle threads

## Original Source

- **Author**: Andrew Vaillancourt
- **Repository**: https://github.com/avcourt/task-parallelism-omp
- **Modified by**: Rafael A. Herrera Guaitero for CARTS compilation

## CARTS Compatibility

- No global variables
- Clean parameter passing
- Command-line argument handling
- OpenMP task parallelism
- Nested parallelism support
- Dynamic thread management

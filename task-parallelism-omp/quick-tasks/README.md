# Quick Tasks - Task-Parallel Quicksort with OpenMP

## Description

A parallel implementation of quicksort using OpenMP task-based parallelism. This benchmark demonstrates how the quicksort divide-and-conquer algorithm can be parallelized by creating independent tasks for partitioned subarrays. Note: This benchmark is from Andrew Vaillancourt's task-parallelism-omp suite.

## Algorithm

**Task-Parallel Quicksort** uses OpenMP tasks to parallelize the recursive partitioning:
1. **Partition**: Select a pivot and partition the array so elements less than the pivot are on the left, greater on the right
2. **Task Creation**: Create two independent OpenMP tasks to sort the left and right partitions
3. **Threshold-based Serial**: When partition size drops below threshold, switch to sequential quicksort to avoid task overhead
4. **Implicit Synchronization**: Tasks complete independently; the calling context waits for both before returning

The algorithm uses `firstprivate` clause to ensure each task has its own copy of necessary variables, enabling safe parallel execution without data races.

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
- **Threshold**: Partition size below which serial quicksort is used (controls task granularity)
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
carts run build/quick-tasks.mlir -- 10000 100 4

# Medium size
carts run build/quick-tasks.mlir -- 100000 500 8

# Large size
carts run build/quick-tasks.mlir -- 1000000 1000 16

# Extra large size
carts run build/quick-tasks.mlir -- 10000000 2000 32
```

### Clean build artifacts

```bash
make clean
```

## Parallelization Strategy

This benchmark uses **OpenMP task-based parallelism** optimized for quicksort's partitioning pattern:

### Task Creation Pattern
```c
#pragma omp parallel
{
  #pragma omp single nowait
  quick_sort(0, n, data, low_limit);
}

void quick_sort(int p, int r, int *data, int low_limit) {
  if (partition_size < low_limit) {
    seq_quick_sort(p, r, data);  // Serial for small partitions
  } else {
    int q = partition(p, r, data);  // Partition once

    #pragma omp task firstprivate(data, low_limit, r, q)
    quick_sort(p, q - 1, data, low_limit);  // Left partition

    #pragma omp task firstprivate(data, low_limit, r, q)
    quick_sort(q + 1, r, data, low_limit);  // Right partition
  }
}
```

### Key Parallelization Features
- **Unbalanced Work Distribution**: Partitioning naturally creates unbalanced subtasks, making dynamic task scheduling essential
- **In-place Partitioning**: Partition operation is sequential but in-place, minimizing memory overhead
- **Task Granularity Control**: Threshold parameter prevents creating too many fine-grained tasks
- **Data Sharing**: Uses `firstprivate` clause to safely share array pointers and indices across tasks
- **Dynamic Load Balancing**: OpenMP runtime assigns tasks to idle threads, handling workload imbalance

### Performance Considerations
- **Pivot Selection**: Uses first element as pivot (can be improved with median-of-three)
- **Partition Imbalance**: Poor pivot selection can create imbalanced partitions, reducing parallelism
- **Threshold Tuning**: Lower threshold increases parallelism but adds task creation overhead
- **Memory Access**: In-place sorting minimizes memory movement, improving cache performance
- **Task Overhead**: Threshold ensures coarse-grained tasks dominate fine-grained overhead

### Comparison with Merge Sort Tasks
- **Space Complexity**: O(log n) stack vs O(n) temporary array for merge sort
- **Load Balancing**: More challenging due to potentially unbalanced partitions
- **Parallelism Depth**: Similar logarithmic depth but varies with pivot quality
- **Cache Performance**: Better locality due to in-place partitioning

## Original Source

- **Author**: Andrew Vaillancourt
- **Repository**: https://github.com/avcourt/task-parallelism-omp
- **Modified by**: Rafael A. Herrera Guaitero for CARTS compilation

## CARTS Compatibility

- No global variables
- Clean parameter passing
- Command-line argument handling
- OpenMP task parallelism
- In-place sorting (minimal memory allocation)
- Proper validation and error checking

# Merge Serial - Sequential Merge Sort Implementation

## Description

A serial implementation of the merge sort algorithm with optimized switching to insertion sort for small subarrays. This benchmark serves as the baseline for comparing against the task-parallel merge sort implementation. Note: This benchmark is from Andrew Vaillancourt's task-parallelism-omp suite.

## Algorithm

**Merge Sort** is a divide-and-conquer sorting algorithm with O(n log n) time complexity:
1. **Divide**: Recursively split the array into two halves until subarrays reach a threshold size
2. **Conquer**: When subarrays are small (≤ 64 elements), switch to insertion sort for efficiency
3. **Merge**: Combine sorted subarrays by comparing elements from each half and copying them in order

This implementation uses a temporary array for merging to avoid excessive memory allocation. The threshold-based switch to insertion sort improves performance on small arrays where merge sort overhead becomes significant.

## Problem Sizes

Note: These benchmarks use runtime parameters rather than compile-time sizes.

| Size | Array Size | Description |
|------|------------|-------------|
| **SMALL** | 10,000 | Small problem size for testing |
| **MEDIUM** | 100,000 | Medium problem size |
| **LARGE** | 1,000,000 | Large problem size |
| **EXTRA LARGE** | 10,000,000 | Very large problem size for performance testing |

The implementation uses a fixed threshold of 64 elements for switching to insertion sort (defined as `SMALL` constant in the code).

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
carts run build/merge-serial.mlir -- 10000

# Medium size
carts run build/merge-serial.mlir -- 100000

# Large size
carts run build/merge-serial.mlir -- 1000000

# Extra large size
carts run build/merge-serial.mlir -- 10000000
```

### Clean build artifacts

```bash
make clean
```

## Implementation Details

The benchmark includes:
- **Merge Sort**: Recursive divide-and-conquer implementation
- **Insertion Sort**: Optimized sorting for small subarrays (≤ 64 elements)
- **Array Initialization**: Random values seeded with 314159 for reproducibility
- **Validation**: Checks that the final array is properly sorted
- **Timing**: Reports execution time in seconds using high-resolution timers

The algorithm switches to insertion sort when subarray size drops to 64 or fewer elements, as insertion sort has lower overhead for small arrays despite its O(n²) complexity.

## Performance Characteristics

- **Time Complexity**: O(n log n) for merge sort, O(n²) for insertion sort on small arrays
- **Space Complexity**: O(n) for temporary merge buffer
- **Stability**: Stable sort (preserves relative order of equal elements)
- **Best/Worst Case**: Consistently O(n log n), making it predictable

## Original Source

- **Author**: Andrew Vaillancourt
- **Repository**: https://github.com/avcourt/task-parallelism-omp
- **Modified by**: Rafael A. Herrera Guaitero for CARTS compilation

## CARTS Compatibility

- No global variables
- Clean parameter passing
- Command-line argument handling
- Dynamic memory allocation
- Proper validation and error checking

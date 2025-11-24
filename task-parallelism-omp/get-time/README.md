# Get Time - High-Resolution Timing Utility

## Description

A portable timing utility that provides high-resolution time measurements using POSIX timers or gettimeofday. This utility is used by other benchmarks in the task-parallelism-omp suite to measure execution times accurately. Note: This benchmark is from Andrew Vaillancourt's task-parallelism-omp suite.

## Algorithm

The utility implements two timing mechanisms with automatic selection at compile time:
- **POSIX Timer Mode**: Uses `clock_gettime()` with `CLOCK_MONOTONIC_RAW` (preferred on Linux 2.6.28+) or `CLOCK_MONOTONIC` for high-resolution, monotonic time measurements
- **gettimeofday Mode**: Falls back to `gettimeofday()` on systems without POSIX timer support

The function returns elapsed time since an arbitrary reference point as a double-precision floating-point value in seconds.

## Problem Sizes

Note: This is a timing utility, not a computational benchmark. It does not process data or have traditional problem sizes.

The timing function itself has negligible overhead (typically sub-microsecond) and is suitable for measuring operations of any duration.

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

### Usage in Applications

This utility is designed to be linked with other benchmarks. It provides a single function:

```c
double get_time(void);
```

Example usage:
```c
double start = get_time();
// ... code to measure ...
double end = get_time();
printf("Elapsed time: %.6f seconds\n", end - start);
```

### Clean build artifacts

```bash
make clean
```

## Implementation Details

The timing utility provides:
- Monotonic time measurements (unaffected by system clock adjustments)
- Nanosecond resolution on systems with POSIX timers
- Microsecond resolution as fallback with gettimeofday
- Automatic error handling with informative messages
- Portable implementation across POSIX-compliant systems

## Original Source

- **Author**: Andrew Vaillancourt
- **Repository**: https://github.com/avcourt/task-parallelism-omp
- **Modified by**: Rafael A. Herrera Guaitero for CARTS compilation

## CARTS Compatibility

- No global variables
- Clean parameter passing
- Reentrant and thread-safe
- Portable across platforms

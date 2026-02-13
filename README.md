# CARTS Benchmark Runner

A powerful CLI tool for building, running, verifying, and reporting on CARTS benchmarks.

## Quick Start

```bash
# List available benchmarks
carts benchmarks list

# Run a single benchmark
carts benchmarks run polybench/gemm --size small --threads 2

# Run with multiple thread counts (thread sweep)
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8

# Run multiple times for statistical significance
carts benchmarks run polybench/gemm --size medium --threads 4 --runs 5

# A JSON results file is always produced (default: carts-benchmarks/results/).
# Use --output to choose a custom location/name.
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 -o results/gemm_scaling.json

# Generate a paper-friendly report (Markdown + SVG figures)
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 --report
```

## Commands

### `carts benchmarks list`

List all available benchmarks.

```bash
carts benchmarks list                    # Show all benchmarks
carts benchmarks list --suite polybench  # Filter by suite
carts benchmarks list --format json      # JSON output
carts benchmarks list --format plain     # Plain list
```

### `carts benchmarks run`

Run benchmarks with verification and timing.

```bash
carts benchmarks run [BENCHMARKS...] [OPTIONS]
```

**Arguments:**
- `BENCHMARKS`: Specific benchmarks to run (optional, runs all if not specified)

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--size` | `-s` | Dataset size: `small`, `medium`, `large` (default: small) |
| `--timeout` | `-t` | Execution timeout in seconds (default: 60) |
| `--threads` | | Thread counts: `1,2,4,8` or `1:16:2` for sweep |
| `--runs` | `-r` | Number of runs per configuration (default: 10) |
| `--omp-threads` | | OpenMP thread count (default: same as ARTS threads) |
| `--launcher` | `-l` | Override ARTS `launcher` (default: from benchmark `arts.cfg`) |
| `--nodes` | `-n` | Override ARTS `nodeCount` (default: from benchmark `arts.cfg`) |
| `--output` | `-o` | Write results JSON to a custom path |
| `--trace` | | Show benchmark output (kernel timing and checksum) |
| `--verbose` | `-v` | Verbose output |
| `--quiet` | `-q` | Minimal output (CI mode) |
| `--no-verify` | | Disable correctness verification |
| `--no-clean` | | Skip cleaning before build (faster, may use stale artifacts) |
| `--debug` | `-d` | Debug level: `0`=off, `1`=commands, `2`=full output to logs |
| `--counter-config` | | Rebuild ARTS with a specific counter profile config file |
| `--counter-dir` | | Directory for ARTS counter output (disabled unless set) |
| `--counters` | | Enable counter collection (auto-creates counter directory) |
| `--cflags` | | Additional CFLAGS: `-DNI=500 -DNJ=500` |
| `--weak-scaling` | | Enable weak scaling (auto-scale problem size) |
| `--base-size` | | Base problem size for weak scaling |
| `--arts-config` | | Custom arts.cfg file |

### `carts benchmarks build`

Build benchmarks without running.

```bash
carts benchmarks build polybench/gemm --size medium
carts benchmarks build --suite polybench --arts   # ARTS only
carts benchmarks build --suite polybench --openmp # OpenMP only
```

### `carts benchmarks clean`

Clean build artifacts.

```bash
carts benchmarks clean polybench/gemm
carts benchmarks clean --all
```

## Usage Examples

### Basic Usage

```bash
# Quick correctness check
carts benchmarks run polybench/gemm --size small --threads 2

# View kernel timing and checksum
carts benchmarks run polybench/gemm --size small --threads 2 --trace
```

### Thread Scaling

```bash
# Strong scaling (fixed problem size, increasing threads)
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8,16 \
    -o results/gemm_strong.json

# Weak scaling (problem size scales with parallelism)
carts benchmarks run polybench/gemm --threads 1,2,4,8 \
    --weak-scaling --base-size 256 \
    -o results/gemm_weak.json
```

### Multiple Runs for Statistics

```bash
# Run 5 times per configuration for statistical significance
carts benchmarks run polybench/gemm --size medium --threads 4 --runs 5 \
    -o results/gemm_stats.json

# Thread sweep with multiple runs
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 --runs 3 \
    -o results/gemm_sweep_stats.json
```

### Debug and Counters

```bash
# Debug level 1: show commands being executed
carts benchmarks run polybench/gemm --size small --threads 2 --debug=1

# Debug level 2: write full output to log files
carts benchmarks run polybench/gemm --size small --threads 2 --debug=2
# Check logs at: polybench/gemm/logs/arts.log and polybench/gemm/logs/omp.log

# Enable ARTS counters with explicit output dir
carts benchmarks run polybench/gemm --size medium --threads 4 \
    --counter-dir results/counters/gemm_4t \
    -o results/gemm_4t.json

# Rebuild ARTS with a custom counter profile, then run
carts benchmarks run polybench/gemm --size medium --threads 4 \
    --counter-config /opt/carts/configs/profile-timing.cfg \
    --counter-dir results/counters/gemm_4t \
    -o results/gemm_4t_profile.json
```

### Custom Problem Sizes

```bash
# Override problem dimensions with CFLAGS
carts benchmarks run polybench/gemm --threads 8 \
    --cflags "-DNI=1024 -DNJ=1024 -DNK=1024" \
    -o results/gemm_1024.json
```

### Suite-wide Execution

```bash
# Run all PolyBench benchmarks
carts benchmarks run --suite polybench --size medium --threads 8

# Run all benchmarks with JSON export
carts benchmarks run --size small --threads 2 -o results/all_benchmarks.json
```

### Different OpenMP Thread Counts

```bash
# Compare ARTS at 4 threads vs OpenMP at 8 threads
carts benchmarks run polybench/gemm --size medium \
    --threads 4 --omp-threads 8 \
    -o results/comparison.json
```

## JSON Output Structure

When using `--output`, results are exported as JSON with the following structure:

```json
{
  "metadata": {
    "timestamp": "2025-12-14T18:22:05",
    "hostname": "arts-node-1",
    "size": "medium",
    "runs_per_config": 3,
    "thread_sweep": [1, 2, 4, 8],
    "launcher": "ssh",
    "artifacts_directory": "gemm_scaling_20251214_182205",
    "reproducibility": {
      "git_commits": { "carts": "abc123", "arts": "def456" },
      "compilers": { "clang": "clang version 18.0.0", ... },
      "cpu": { "cores": 16 },
      "system": { "os": "Linux", ... }
    }
  },
  "summary": {
    "total_configs": 4,
    "total_runs": 12,
    "passed": 12,
    "failed": 0,
    "skipped": 0,
    "pass_rate": 1.0,
    "avg_speedup": 1.05,
    "geometric_mean_speedup": 1.03,
    "statistics": {
      "4_threads": {
        "arts_kernel_time": { "mean": 0.0234, "stddev": 0.0012, "min": 0.022, "max": 0.025 },
        "omp_kernel_time": { "mean": 0.0256, "stddev": 0.0008, "min": 0.024, "max": 0.027 },
        "speedup": { "mean": 1.09, "stddev": 0.02, "min": 1.06, "max": 1.12 },
        "run_count": 3
      }
    }
  },
  "results": [
    {
      "name": "polybench/gemm",
      "suite": "polybench",
      "size": "medium",
      "config": {
        "arts_threads": 4,
        "arts_nodes": 1,
        "omp_threads": 4,
        "launcher": "ssh"
      },
      "run_number": 1,
      "build_arts": { "status": "pass", "duration_sec": 0.20 },
      "build_omp": { "status": "pass", "duration_sec": 0.001 },
      "run_arts": {
        "status": "pass",
        "duration_sec": 0.42,
        "exit_code": 0,
        "checksum": "1.288433871069e+06",
        "kernel_timings": { "gemm": 0.0234 }
      },
      "run_omp": {
        "status": "pass",
        "duration_sec": 0.10,
        "exit_code": 0,
        "checksum": "1.288433813335e+06",
        "kernel_timings": { "gemm": 0.0256 }
      },
      "timing": {
        "arts_kernel_sec": 0.0234,
        "omp_kernel_sec": 0.0256,
        "speedup": 1.09
      },
      "verification": {
        "correct": true,
        "tolerance": 0.01,
        "note": "Checksums match within tolerance"
      },
      "timestamp": "2025-12-14T18:22:01"
    }
  ],
  "failures": []
}
```

## ARTS Configuration Override

ARTS uses a three-tier configuration fallback system when no explicit config is specified:

### Configuration Discovery Priority

1. **Custom config** (`--arts-config /path/to/config.cfg`)
2. **Local config** (`benchmark_dir/arts.cfg`)
3. **Global default config** (`carts-benchmarks/arts.cfg`)

If no `--arts-config` is provided, CARTS first looks for an `arts.cfg` file in the benchmark directory. If that doesn't exist, it falls back to the global default configuration.

The runner displays the effective ARTS configuration before execution:
- Single benchmark: shows specific config values (threads, nodes, launcher)
- Multiple benchmarks without `--arts-config`: shows "ARTS Config: using local"
- Multiple benchmarks with `--arts-config`: shows specific custom config values

### Command-Line Overrides

```bash
# Override launcher and node count
carts benchmarks run polybench/gemm --launcher slurm --nodes 4

# Override thread count and launcher
carts benchmarks run polybench/gemm --threads 32 --launcher ssh --nodes 2

# Override OpenMP thread count separately
carts benchmarks run polybench/gemm --threads 16 --omp-threads 8
```

### Custom Configuration Files

```bash
# Use a completely custom arts.cfg file
carts benchmarks run polybench/gemm --arts-config /path/to/my_config.cfg

# Example custom config for multi-node execution
echo -e "[ARTS]\nthreads=64\nlauncher=slurm\nnodeCount=4\nnodes=node001,node002,node003,node004" > multi.cfg
carts benchmarks run polybench/gemm --arts-config multi.cfg
```

### Overrideable Parameters

| Parameter | CLI Option | Description |
|-----------|------------|-------------|
| `launcher` | `--launcher` | Job launcher (ssh, slurm, lsf) |
| `nodeCount` | `--nodes`, `-n` | Number of compute nodes |
| `threads` | `--threads` | ARTS worker threads per node |
| `omp-threads` | `--omp-threads` | OpenMP threads (separate from ARTS threads) |

Command-line options take precedence over any configuration file settings.

### Key Fields

| Field | Description |
|-------|-------------|
| `metadata.runs_per_config` | Number of times each configuration was run |
| `metadata.thread_sweep` | List of thread counts tested |
| `metadata.artifacts_directory` | "." (artifacts are in same folder as JSON) |
| `summary.total_configs` | Number of unique (benchmark, threads, nodes) configurations |
| `summary.total_runs` | Total number of benchmark executions |
| `summary.statistics` | Per-config statistics (only when `--runs > 1`) |
| `results[].config` | Configuration for this specific run |
| `results[].run_number` | Which iteration this is (1-N) |
| `results[].timing.speedup` | OMP kernel time / ARTS kernel time (>1 = ARTS faster) |

## Experiment Organization

When using `--output`, results are placed in a self-contained timestamped experiment directory. The `-o` argument specifies the **base name** (without `.json` extension).

### Basic Usage

```bash
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 \
    -o results/gemm_scaling
```

Creates:
```
results/gemm_scaling_20251214_182205/       # Self-contained experiment folder
├── gemm_scaling.json                       # Results JSON (inside folder)
└── polybench/gemm/build/                   # Build artifacts
    ├── 1t_1n/
    ├── 2t_1n/
    ├── 4t_1n/
    └── 8t_1n/
```

### Running Multiple Experiments

Organize experiments under a common parent directory:

```bash
# All experiments go under results/single_rank/
carts benchmarks run --size small --threads 2 -o results/single_rank/correctness_2t
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 -o results/single_rank/gemm_strong
carts benchmarks run polybench/jacobi2d --size medium --threads 1,2,4,8 -o results/single_rank/jacobi_strong
```

Result:
```
results/single_rank/
├── correctness_2t_20251214_183000/
│   ├── correctness_2t.json
│   └── polybench/...
├── gemm_strong_20251214_183100/
│   ├── gemm_strong.json
│   └── polybench/gemm/build/...
└── jacobi_strong_20251214_183200/
    ├── jacobi_strong.json
    └── polybench/jacobi2d/build/...
```

### Full Directory Structure

Each experiment folder contains both results and artifacts:

```
results/gemm_scaling_20251214_182205/
├── gemm_scaling.json                       # Results JSON
└── polybench/                              # Suite folder
    └── gemm/                               # Benchmark folder
        └── build/                          # All builds for this benchmark
            ├── 2t_1n/                      # Config: 2 threads, 1 node
            │   ├── arts.cfg                # Build configuration (INPUT)
            │   ├── artifacts/              # Build outputs
            │   │   ├── .carts-metadata.json    # Compiler analysis
            │   │   ├── gemm_arts_metadata.mlir # MLIR with metadata
            │   │   ├── gemm.mlir               # Parallel MLIR
            │   │   ├── gemm-arts.ll            # LLVM IR
            │   │   ├── gemm_arts               # ARTS executable
            │   │   ├── gemm_omp                # OpenMP executable
            │   │   ├── build_arts.log          # ARTS build log
            │   │   └── build_openmp.log        # OpenMP build log
            │   └── runs/                   # Execution outputs
            │       ├── 1/                  # First run
            │       │   ├── arts.log        # ARTS runtime output
            │       │   ├── omp.log         # OpenMP runtime output
            │       │   └── counters/       # Counter files (if --counters)
            │       ├── 2/                  # Second run
            │       └── 3/                  # Third run
            │
            ├── 4t_1n/                      # Config: 4 threads, 1 node
            │   ├── arts.cfg
            │   ├── artifacts/
            │   └── runs/
            │
            └── 8t_1n/                      # Config: 8 threads, 1 node
                └── ...
```

### Key Concepts

1. **Self-contained experiments**: Each experiment folder contains both the JSON results and all artifacts, making it portable and easy to share.

2. **Timestamped directories**: Each experiment creates a unique timestamped folder (`{name}_{YYYYMMDD_HHMMSS}/`), so multiple runs don't overwrite each other.

3. **Build vs Run artifacts**: Changing threads or nodes requires recompilation. Each unique config (`{threads}t_{nodes}n/`) has its own build artifacts that are shared across multiple runs.

4. **Finding all results**: Use `find results/ -name "*.json"` to discover all experiment results.

5. **Compiler metadata**: `.carts-metadata.json` contains loop and memory reference analysis from the CARTS compiler, useful for understanding parallelization decisions.

### CLI Output

When export is enabled, the CLI shows the experiment folder:

```
✓ polybench/gemm [4 threads] PASS (speedup: 1.09x)

Experiment folder: results/gemm_scaling_20251214_182205
Results JSON:      results/gemm_scaling_20251214_182205/gemm_scaling.json
```

## Artifact Output

Every run persists logs and results to disk, even without `--output`. The default
results directory is `./results`.

### Directory Structure

```
results/
  20260212_143052/                        # auto-created every run
    polybench_gemm/
      build/
        2t_1n/
          arts.cfg                        # build configuration
          artifacts/                      # build outputs (MLIR, LLVM IR, executables)
          runs/
            1/
              arts.log                    # ARTS runtime stdout/stderr
              omp.log                     # OpenMP runtime stdout/stderr
              counters/                   # counter files (if --counters)
            2/
    results.json                          # (only with --output)
```

When `--output` is specified, the experiment directory also contains build artifacts
and the results JSON. Without `--output`, a timestamped directory is still created
under `./results` for log persistence.

### After a run, the CLI prints an artifact summary:

```
Artifacts:
  Results:  ./results/20260212_143052.json
  Logs:     ./results/20260212_143052/polybench_gemm/build/2t_1n/runs/1/
  Counters: ./results/counters/  (if --counters)
```

## Counters

ARTS counters collect per-EDT timing and invocation metrics.

### Quick start

```bash
# Simple: --counters auto-creates the counter directory
carts benchmarks run polybench/gemm --size small --threads 2 --counters

# Explicit: specify a custom counter directory
carts benchmarks run polybench/gemm --size small --threads 2 \
    --counter-dir results/counters/gemm_2t
```

`--counters` is sugar for `--counter-dir ./results/counters`. Both enable counter
collection; `--counter-dir` lets you choose the output location.

### Counter file contents

Counter output is a `cluster.json` file (or per-thread `n0_t0.json`, `n0_t1.json`, etc.)
containing EDT (Event-Driven Task) metrics:

- `artsIdMetrics.edts[].total_exec_ns` - execution time per arts_id
- `artsIdMetrics.edts[].invocations` - how often each EDT runs
- `artsIdMetrics.edts[].total_stall_ns` - time waiting for data

### Custom counter profiles

Counter files require ARTS to be built with counter support.
Use `--counter-config` to rebuild ARTS with a specific profile before running:

```bash
carts benchmarks run polybench/gemm --size medium --threads 4 \
    --counter-config /opt/carts/configs/profile-timing.cfg \
    --counters
```

## Debugging a Benchmark

### Log files

Logs are always written to disk. For thread sweep mode, they go to the experiment
directory. For standard (multi-benchmark) mode, they go to `{benchmark}/logs/`.

Each log file contains:
- The exact command that was executed
- Execution duration and exit code
- Full stdout and stderr

### Debug levels

```bash
# Level 1: print commands being executed to the console
carts benchmarks run polybench/gemm --size small --threads 2 --debug=1

# Level 2: also print log file paths to the console
carts benchmarks run polybench/gemm --size small --threads 2 --debug=2
```

### Inspecting MLIR stages

To see the MLIR output at each compiler stage, use `--pipeline` to stop at a
specific point:

```bash
# Stop after the concurrency pass (parallel MLIR)
carts compile polybench/gemm/gemm.c --pipeline concurrency

# Run carts-compile directly for clean MLIR output
/opt/carts/.install/carts/bin/carts-compile gemm.mlir --O3 --arts-config arts.cfg --concurrency
```

## Troubleshooting

### Port conflicts

ARTS uses TCP port 34739 by default. If a previous run left lingering processes,
the next run may fail with a port-in-use error. The runner automatically kills
processes on the ARTS port before each run. To manually clear:

```bash
fuser -k 34739/tcp
```

### Stale processes

For SSH multi-node runs, the runner cleans up stale ARTS processes before launch.
If you see issues, manually kill on all nodes:

```bash
pkill -f '_arts'
```

### Missing checksums

If a benchmark reports no checksum, check:
1. The benchmark prints a line matching `checksum: <value>` on stdout
2. The run didn't crash (check the log file for stderr)
3. The timeout wasn't exceeded (`--timeout` defaults to 60s)

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All benchmarks passed |
| 1 | One or more benchmarks failed or crashed |

## Available Benchmark Suites

### PolyBench Suite

Linear algebra kernels and stencil computations from PolyBench/C.

- `polybench/gemm` - Matrix multiplication (O(N^3) compute-bound)
- `polybench/2mm`, `polybench/3mm` - Multiple matrix operations
- `polybench/jacobi2d` - 2D Jacobi stencil (memory-bound)
- `polybench/heat-3d` - 3D heat equation
- And more...

### KaStORS Suite

Task-based parallel benchmarks for OpenMP task dependencies.

- `kastors-jacobi/jacobi-task-dep` - Task dependency Jacobi
- `kastors-jacobi/jacobi-for` - Fork-join Jacobi
- `kastors-jacobi/jacobi-block-for` - Blocked fork-join Jacobi

Use `carts benchmarks list` to see all available benchmarks.

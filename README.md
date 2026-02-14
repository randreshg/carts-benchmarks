# CARTS Benchmark Runner

A powerful CLI tool for building, running, and verifying CARTS benchmarks.

## Quick Start

```bash
# List available benchmarks
carts benchmarks list

# Run a single benchmark (results in carts-benchmarks/results/{timestamp}/)
carts benchmarks run polybench/gemm --size small --threads 2

# Run with multiple thread counts (thread sweep)
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8

# Node sweep (tests each thread count at each node count)
carts benchmarks run polybench/gemm --threads 1,2,4 --nodes 1,2

# Run multiple times for statistical significance
carts benchmarks run polybench/gemm --size medium --threads 4 --runs 5

# Custom results directory
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 --results-dir results/scaling

# Analyze results after a run
carts analyze summary results/20240115_120530/
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
| `--nodes` | `-n` | Node counts: single (`2`), list (`1,2,4`), range (`1:8:2`) |
| `--results-dir` | | Base directory for experiment output (default: `carts-benchmarks/results/`) |
| `--trace` | | Show benchmark output (kernel timing and checksum) |
| `--verbose` | `-v` | Verbose output |
| `--quiet` | `-q` | Minimal output (CI mode) |
| `--no-verify` | | Disable correctness verification |
| `--no-clean` | | Skip cleaning before build (faster, may use stale artifacts) |
| `--debug` | `-d` | Debug level: `0`=off, `1`=commands, `2`=verbose console output (logs always captured) |
| `--counters` | | Counter level: `0`=off (default), `1`=artsid metrics, `2`=deep captures |
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

## Analyzing Results

After running benchmarks, use `carts analyze` to examine results.

### `carts analyze summary`

Re-display the results table from a completed experiment.

```bash
carts analyze summary results/20240115_120530/
carts analyze summary results/20240115_120530/ -b gemm    # filter by benchmark
carts analyze summary results/20240115_120530/ --sort speedup
```

### `carts analyze export`

Export timing data to CSV for external tools (spreadsheets, notebooks).

```bash
# Export to file
carts analyze export results/20240115_120530/ -o timing.csv

# Export to stdout (pipe to other tools)
carts analyze export results/20240115_120530/ | head -5

# Filter by benchmark
carts analyze export results/20240115_120530/ -b gemm -o gemm.csv
```

**CSV columns:** `benchmark, threads, nodes, run, arts_e2e_sec, omp_e2e_sec, arts_kernel_sec, omp_kernel_sec, arts_init_sec, omp_init_sec, speedup, status`

### `carts analyze compare`

Compare two experiments side-by-side to identify improvements and regressions.

```bash
# Compare baseline vs optimized
carts analyze compare results/baseline/ results/optimized/

# With custom threshold (flag changes > 10%)
carts analyze compare results/before/ results/after/ --threshold 0.10
```

Output shows per-benchmark speedup delta with IMPROVED/REGRESSED/SAME verdicts.

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
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8,16

# Weak scaling (problem size scales with parallelism)
carts benchmarks run polybench/gemm --threads 1,2,4,8 \
    --weak-scaling --base-size 256
```

### Node Scaling

```bash
# Single node count override
carts benchmarks run polybench/gemm --size medium --threads 4 --nodes 2

# Node sweep (tests default threads at each node count)
carts benchmarks run polybench/gemm --nodes 1,2,4

# 2D sweep: thread x node (4 configs: 1t_1n, 2t_1n, 1t_2n, 2t_2n)
carts benchmarks run polybench/gemm --threads 1,2 --nodes 1,2
```

### Multiple Runs for Statistics

```bash
# Run 5 times per configuration for statistical significance
carts benchmarks run polybench/gemm --size medium --threads 4 --runs 5

# Thread sweep with multiple runs
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 --runs 3
```

### Debug and Counters

```bash
# Debug level 1: show commands being executed
carts benchmarks run polybench/gemm --size small --threads 2 --debug=1

# Debug level 2: verbose console output (logs are always captured)
carts benchmarks run polybench/gemm --size small --threads 2 --debug=2

# Collect ARTS counters (requires ARTS built with counters enabled)
carts benchmarks run polybench/gemm --size medium --threads 4 --counters=1
```

### Custom Problem Sizes

```bash
# Override problem dimensions with CFLAGS
carts benchmarks run polybench/gemm --threads 8 \
    --cflags "-DNI=1024 -DNJ=1024 -DNK=1024"
```

### Suite-wide Execution

```bash
# Run all PolyBench benchmarks
carts benchmarks run --suite polybench --size medium --threads 8

# Run all benchmarks
carts benchmarks run --size small --threads 2
```

### Different OpenMP Thread Counts

```bash
# Compare ARTS at 4 threads vs OpenMP at 8 threads
carts benchmarks run polybench/gemm --size medium \
    --threads 4 --omp-threads 8
```

## JSON Output Structure

Every run produces a `results.json` inside the experiment directory with the following structure:

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
| `nodeCount` | `--nodes`, `-n` | Number of compute nodes (supports sweep) |
| `threads` | `--threads` | ARTS worker threads per node |
| `omp-threads` | `--omp-threads` | OpenMP threads (separate from ARTS threads) |

Command-line options take precedence over any configuration file settings.

### Key Fields

| Field | Description |
|-------|-------------|
| `metadata.runs_per_config` | Number of times each configuration was run |
| `metadata.thread_sweep` | List of thread counts tested |
| `summary.total_configs` | Number of unique (benchmark, threads, nodes) configurations |
| `summary.total_runs` | Total number of benchmark executions |
| `summary.statistics` | Per-config statistics (only when `--runs > 1`) |
| `results[].config` | Configuration for this specific run |
| `results[].run_number` | Which iteration this is (1-N) |
| `results[].timing.speedup` | OMP kernel time / ARTS kernel time (>1 = ARTS faster) |

## Experiment Output

Every `carts benchmarks run` invocation creates a self-contained timestamped
experiment directory under `carts-benchmarks/results/`:

### Directory Structure

```
results/{YYYYMMDD_HHMMSS}/
├── manifest.json                        # Structure index + quick summary
├── results.json                         # Full results data
└── polybench/gemm/                      # Per-benchmark
    └── 4t_1n/                           # Per-config (threads x nodes)
        ├── artifacts/                   # Build outputs (shared across runs)
        │   ├── arts.cfg
        │   ├── .carts-metadata.json
        │   ├── gemm_arts_metadata.mlir
        │   ├── gemm.mlir
        │   ├── gemm-arts.ll
        │   ├── gemm_arts, gemm_omp
        │   └── build_arts.log, build_omp.log
        ├── run_1/                       # Per-run outputs
        │   ├── arts.log                 # ARTS stdout+stderr (always captured)
        │   ├── omp.log                  # OpenMP stdout+stderr
        │   ├── counters/                # If counters enabled
        │   │   └── cluster.json
        │   └── perf/                    # If --perf enabled
        │       ├── arts_cache.csv
        │       └── omp_cache.csv
        └── run_2/                       # If --runs > 1
            └── ...
```

### Key Properties

1. **Self-contained**: Everything in one directory — portable, easy to share/archive.

2. **Always captured**: stdout/stderr saved to `arts.log`/`omp.log` in every run (no `--debug` required).

3. **Manifest**: `manifest.json` describes the layout for analysis tools. All paths are relative to the experiment directory, so the folder can be moved or tarred.

4. **Timestamped**: Each run creates a unique `{YYYYMMDD_HHMMSS}/` directory, so multiple runs never overwrite each other.

5. **Build vs Run artifacts**: Each unique config (`{threads}t_{nodes}n/`) has its own build artifacts in `artifacts/` that are shared across multiple runs.

6. **Compiler metadata**: `.carts-metadata.json` contains loop and memory reference analysis from the CARTS compiler.

### Custom Results Directory

```bash
# Override the default results location
carts benchmarks run polybench/gemm --size medium --threads 1,2,4,8 \
    --results-dir results/scaling
```

### CLI Output

```
✓ polybench/gemm [4 threads] PASS (speedup: 1.09x)

Results: carts-benchmarks/results/20251214_182205
```

## Counter Files

When ARTS is built with counter support, counter data is automatically saved
inside each run directory:

```
run_1/counters/
  cluster.json    # Cluster-wide counter data (init time, e2e time)
  n0_t0.json      # Per-thread counter data
  ...
```

Counter files contain EDT (Event-Driven Task) metrics:
- `artsIdMetrics.edts[].total_exec_ns` - execution time per arts_id
- `artsIdMetrics.edts[].invocations` - how often each EDT runs
- `artsIdMetrics.edts[].total_stall_ns` - time waiting for data

**Note**: Counter files require ARTS to be rebuilt with counter support:
```bash
carts build --arts --counters=1  # Level 1: ArtsID metrics
carts build --arts --counters=2  # Level 2: Deep captures
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

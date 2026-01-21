# Graph500 Graph-Gen Analysis

## Summary

This benchmark allocates each adjacency list inside the parallel loop to
exercise distributed allocation. Canonicalize-memrefs now recognizes
array-of-arrays initialization even when the store happens inside an
`omp.wsloop`, so the parallel allocation pattern stays intact.

## Build and Run

```bash
cd /Users/randreshg/Documents/carts
./tools/carts build

cd external/carts-benchmarks/graph500/graph-gen
make small
./graph_gen_arts
./build/graph_gen_omp
```

Or via the benchmark runner:

```bash
./tools/carts benchmarks run graph500/graph-gen --size small
```

## Expected Behavior

- ARTS and OpenMP should print the same `checksum` (total edges).
- Memory footprint scales with `SCALE * EDGE_FACTOR`.

## Notes

- `arts.cfg` is sourced from `external/carts-benchmarks/arts.cfg` when the
  local suite does not provide one.
- Edge counts are summed after the parallel region to avoid OpenMP
  reductions in the compiler pipeline.

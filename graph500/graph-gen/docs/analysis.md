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

---

<!-- BEGIN DISTRIBUTION DIAGRAMS -->
## Distribution Diagrams

These diagrams show how CARTS/ARTS distribute work and datablocks for this
example when internode routing is enabled.

### 1) Work Routing (ForLowering)

```mermaid
flowchart LR
  B["Chunk/Block index b"] --> C["chunksPerWorker = ceil(totalChunks / (totalNodes * workersPerNode))"]
  C --> W["worker = floor(b / chunksPerWorker)"]
  W --> R["route = worker % totalNodes"]
  R --> T["EDT is launched on route node"]
```

### 2) Distributed DB Ownership

```mermaid
flowchart LR
  DB["DB block index b"] --> W2["worker = floor(b / chunksPerWorker)"]
  W2 --> R2["ownerRoute = worker % totalNodes"]
  R2 --> G["artsReserveGuidRoute(dbMode, ownerRoute)"]
  G --> O["Owner node creates local DB payload"]
```

### 3) Host-Init Read-Only DB Flush Path

```mermaid
sequenceDiagram
  participant H as Rank 0 host-init
  participant O as Owner node (route r)
  H->>H: Initialize local shadow buffer
  H->>O: artsPutInDbEpoch(payload, epoch, guid[r], ...)
  O->>O: Commit owner-local DB payload
```

### 4) Verification Commands

```bash
# Task routing + distributed markers
carts run <example>.mlir --concurrency --debug-only=for_lowering 2>&1 | \
  rg "route|worker|distributed"

# Partitioning/full-range decisions
carts run <example>.mlir --concurrency-opt --debug-only=db,db_partitioning 2>&1 | \
  rg "partition|full-range|mode"

# LLVM/runtime ownership calls
rg -n "initPerNode|artsReserveGuidRoute|artsDbCreateRemote|artsPutInDbEpoch" \
  <example>-arts.ll
```

Notes:
- `READ` acquire means task must not modify payload.
- `WRITE` acquire means task may modify payload.
- Mutable host-store+host-load allocations are currently
  kept local (no distributed host-readback path is emitted).
<!-- END DISTRIBUTION DIAGRAMS -->

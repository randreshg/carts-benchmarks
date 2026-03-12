/**
 * Graph500-Inspired Graph Generation Benchmark
 *
 * This benchmark demonstrates the scalability wall pattern:
 * - Each vertex's adjacency list is allocated INSIDE the parallel loop
 * - OpenMP: All allocations contend for single-node memory
 * - CARTS: Allocations are distributed across nodes
 *
 * At scale 32 (2^32 vertices), the graph structure exceeds single-node memory.
 * OpenMP fails with OOM; CARTS succeeds by distributing across nodes.
 */

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "arts/utils/benchmarks/CartsBenchmarks.h"

#ifndef SCALE
#define SCALE 20  /* 2^SCALE vertices */
#endif

#ifndef EDGE_FACTOR
#define EDGE_FACTOR 16  /* Average edges per vertex (Graph500 default) */
#endif

/* Simple PRNG for reproducible edge generation */
static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/* Generate target vertex using R-MAT-like distribution */
static inline uint64_t rmat_target(uint64_t src, uint64_t num_vertices, uint64_t edge_idx) {
    uint64_t state = src * 0x123456789abcdef + edge_idx;
    return splitmix64(&state) % num_vertices;
}

int main(int argc, char *argv[]) {
    CARTS_BENCHMARKS_START();
    CARTS_E2E_TIMER_START("graph_gen");
    CARTS_STARTUP_TIMER_START("graph_gen");

    uint64_t num_vertices = (uint64_t)1 << SCALE;
    uint64_t edges_per_vertex = EDGE_FACTOR;
    /* Allocate outer arrays (vertex pointers and counts) */
    uint64_t **adj_list = (uint64_t **)malloc(num_vertices * sizeof(uint64_t *));
    uint64_t *adj_count = (uint64_t *)malloc(num_vertices * sizeof(uint64_t));
    CARTS_STARTUP_TIMER_STOP();

    CARTS_KERNEL_TIMER_START("graph_gen");

    /* Generate edges - allocate adjacency list INSIDE loop */
    #pragma omp parallel for schedule(dynamic)
    for (uint64_t v = 0; v < num_vertices; v++) {
        adj_list[v] = (uint64_t *)malloc(edges_per_vertex * sizeof(uint64_t));
        adj_count[v] = 0;

        /* Generate edges using R-MAT-like distribution */
        for (uint64_t e = 0; e < edges_per_vertex; e++) {
            uint64_t target = rmat_target(v, num_vertices, e);
            adj_list[v][adj_count[v]++] = target;
        }
    }

    CARTS_KERNEL_TIMER_STOP("graph_gen");

    CARTS_VERIFICATION_TIMER_START("graph_gen");
    /* Total edges generated (for checksum) */
    uint64_t total_edges = 0;
    for (uint64_t v = 0; v < num_vertices; v++) {
        total_edges += adj_count[v];
    }
    double checksum = (double)total_edges;
    CARTS_BENCH_CHECKSUM(checksum);
    CARTS_VERIFICATION_TIMER_STOP();

    CARTS_CLEANUP_TIMER_START("graph_gen");
    /* Free memory */
    for (uint64_t v = 0; v < num_vertices; v++) {
        free(adj_list[v]);
    }
    free(adj_list);
    free(adj_count);
    CARTS_CLEANUP_TIMER_STOP();

    CARTS_E2E_TIMER_STOP();
    CARTS_BENCHMARKS_STOP();

    return 0;
}

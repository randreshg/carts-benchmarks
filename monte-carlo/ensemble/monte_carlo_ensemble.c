/**
 * Monte Carlo Ensemble Benchmark
 *
 * This benchmark demonstrates the scalability wall pattern:
 * - Each sample allocates a state matrix INSIDE the parallel loop
 * - OpenMP: All samples contend for single-node memory
 * - CARTS: Each sample allocates on its executing node
 *
 * At 100,000 samples (~100 GB total), OpenMP exhausts single-node memory.
 * CARTS distributes samples across nodes, each using local memory allocation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"

#ifndef NUM_SAMPLES
#define NUM_SAMPLES 1000  /* Number of Monte Carlo samples */
#endif

#ifndef STATE_DIM
#define STATE_DIM 1024  /* State matrix dimension (STATE_DIM x STATE_DIM) */
#endif

/* Simple computation function for each sample */
static double compute_sample_value(unsigned sample_idx, unsigned i, unsigned j) {
    /* Pseudo-random computation based on indices */
    double val = sin((double)(sample_idx * 17 + i * 31 + j * 7) * 0.001);
    return val * val;
}

/* Reduce a state matrix to a single value */
static double reduce_state(double **state, unsigned dim) {
    double sum = 0.0;
    for (unsigned i = 0; i < dim; i++) {
        for (unsigned j = 0; j < dim; j++) {
            sum += state[i][j];
        }
    }
    return sum / (dim * dim);
}

/* Free a state matrix */
static void free_state(double **state, unsigned dim) {
    for (unsigned i = 0; i < dim; i++) {
        free(state[i]);
    }
    free(state);
}

int main(int argc, char *argv[]) {
    CARTS_BENCHMARKS_START();

    unsigned num_samples = NUM_SAMPLES;
    unsigned state_dim = STATE_DIM;
    double memory_per_sample = (double)(state_dim * state_dim * sizeof(double)) / (1024.0 * 1024.0);
    double total_memory = memory_per_sample * num_samples / 1024.0;

    printf("Monte Carlo Ensemble: samples=%u, state_dim=%u\n", num_samples, state_dim);
    printf("Memory per sample: %.2f MB, Total: %.2f GB\n", memory_per_sample, total_memory);

    CARTS_E2E_TIMER_START("monte_carlo_ensemble");

    double global_sum = 0.0;
    double *sample_sums = (double *)malloc(num_samples * sizeof(double));
    if (!sample_sums) {
        fprintf(stderr, "Failed to allocate sample_sums\n");
        return 1;
    }

    CARTS_KERNEL_TIMER_START("parallel_samples");

    /* Process samples in parallel - state allocated INSIDE loop */
    /* This is the key pattern for distributed scaling */
    #pragma omp parallel for schedule(dynamic)
    for (unsigned s = 0; s < num_samples; s++) {
        /* Allocate sample state INSIDE loop (distributed by CARTS) */
        /* Using array-of-arrays pattern required by CARTS */
        double **state = (double **)malloc(state_dim * sizeof(double*));
        for (unsigned i = 0; i < state_dim; i++) {
            state[i] = (double *)malloc(state_dim * sizeof(double));
        }

        /* Initialize and compute sample */
        for (unsigned i = 0; i < state_dim; i++) {
            for (unsigned j = 0; j < state_dim; j++) {
                state[i][j] = compute_sample_value(s, i, j);
            }
        }

        /* Reduce sample result and accumulate */
        double sample_result = reduce_state(state, state_dim);
        sample_sums[s] = sample_result;

        /* Free local state */
        free_state(state, state_dim);
    }

    CARTS_KERNEL_TIMER_STOP("parallel_samples");

    for (unsigned s = 0; s < num_samples; s++) {
        global_sum += sample_sums[s];
    }
    free(sample_sums);

    CARTS_E2E_TIMER_STOP();

    /* Compute checksum */
    CARTS_BENCH_CHECKSUM(global_sum);

    printf("Global sum: %.6f (average per sample: %.6f)\n",
           global_sum, global_sum / num_samples);

    return 0;
}

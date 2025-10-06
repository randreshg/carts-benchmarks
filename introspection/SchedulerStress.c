#include <inttypes.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  unsigned depth;
  unsigned fanout;
  unsigned spinIters;
} BenchConfig;

typedef struct {
  uint64_t tasksExecuted;
} BenchResults;

static void spawnTasks(unsigned depth, const BenchConfig *config,
                       BenchResults *results) {
  // Inline busyWork function
  volatile double acc = 0.0;
  for (unsigned i = 0; i < config->spinIters; ++i)
    acc += (double)(i + 1) * 0.61803398875;
  (void)acc;

#pragma omp atomic
  results->tasksExecuted++;
  if (depth == 0)
    return;
  for (unsigned i = 0; i < config->fanout; ++i) {
#pragma omp task firstprivate(depth)
    spawnTasks(depth - 1, config, results);
  }
#pragma omp taskwait
}

int main(int argc, char **argv) {
  // Initialize benchmark configuration
  BenchConfig config = {.depth = 5, .fanout = 4, .spinIters = 512};

  // Inline parseArgs function
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth") && i + 1 < argc) {
      config.depth = (unsigned)strtoul(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) {
      config.fanout = (unsigned)strtoul(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--spin") && i + 1 < argc) {
      config.spinIters = (unsigned)strtoul(argv[++i], NULL, 10);
    }
  }
  if (config.fanout < 1)
    config.fanout = 1;
  if (!config.spinIters)
    config.spinIters = 1;

  // Inline computeTotalTasks function
  uint64_t expected = 0;
  uint64_t level = 1;
  for (unsigned i = 0; i <= config.depth; ++i) {
    expected += level;
    level *= config.fanout;
    if (!config.fanout)
      break;
  }

  // Initialize benchmark results
  BenchResults results = {.tasksExecuted = 0};

  double benchStart = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    spawnTasks(config.depth, &config, &results);
  }
  double elapsed = omp_get_wtime() - benchStart;

  printf("BENCH:scheduler elapsed_s=%.6f tasks=%" PRIu64 " expected=%" PRIu64
         " depth=%u fanout=%u spin=%u\n",
         elapsed, results.tasksExecuted, expected, config.depth, config.fanout,
         config.spinIters);
  if (results.tasksExecuted != expected) {
    fprintf(stderr,
            "warning: expected %" PRIu64 " tasks but observed %" PRIu64 "\n",
            expected, results.tasksExecuted);
  }
  fflush(stdout);
  return 0;
}

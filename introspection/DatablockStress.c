#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  unsigned depth;
  unsigned fanout;
  size_t dbSizeBytes;
  size_t opsPerTask;
} BenchConfig;

typedef struct {
  uint64_t tasksExecuted;
  uint64_t totalBytesMoved;
} BenchResults;

static void spawnTasks(unsigned depth, const BenchConfig *config,
                       BenchResults *results) {
  for (size_t i = 0; i < config->opsPerTask; ++i) {
    void *buffer = malloc(config->dbSizeBytes);
    if (buffer)
      memset(buffer, (int)(i & 0xff), config->dbSizeBytes);
#pragma omp atomic
    results->totalBytesMoved += (uint64_t)config->dbSizeBytes;
    free(buffer);
  }

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
  BenchConfig config = {
      .depth = 3, .fanout = 4, .dbSizeBytes = 1024, .opsPerTask = 64};

  // Inline parseArgs function
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth") && i + 1 < argc) {
      config.depth = (unsigned)strtoul(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) {
      config.fanout = (unsigned)strtoul(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--ops") && i + 1 < argc) {
      config.opsPerTask = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--db-size") && i + 1 < argc) {
      config.dbSizeBytes = (size_t)strtoull(argv[++i], NULL, 10);
    }
  }
  if (config.fanout < 1)
    config.fanout = 1;
  if (!config.opsPerTask)
    config.opsPerTask = 1;
  if (!config.dbSizeBytes)
    config.dbSizeBytes = 1;

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
  BenchResults results = {.tasksExecuted = 0, .totalBytesMoved = 0};

  double start = omp_get_wtime();

#pragma omp parallel
  {
#pragma omp single
    spawnTasks(config.depth, &config, &results);
  }

  double elapsed = omp_get_wtime() - start;
  const uint64_t totalOps = expected * config.opsPerTask;
  printf("BENCH:datablock elapsed_s=%.6f tasks=%" PRIu64 " expected=%" PRIu64
         " total_ops=%" PRIu64 " bytes=%" PRIu64 " depth=%u fanout=%u\n",
         elapsed, results.tasksExecuted, expected, totalOps,
         (uint64_t)(results.totalBytesMoved), config.depth, config.fanout);
  if (results.tasksExecuted != expected) {
    fprintf(stderr,
            "warning: expected %" PRIu64 " tasks but observed %" PRIu64 "\n",
            expected, results.tasksExecuted);
  }
  fflush(stdout);
  return 0;
}

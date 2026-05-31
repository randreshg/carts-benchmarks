################################################################################
# PolyBench specific defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

POLYBENCH_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(POLYBENCH_ROOT)/..)

# Set includes for common/carts.mk
INCLUDES ?= -I. -I../common -I../utilities
MINI_CFLAGS ?= -DMINI_DATASET
STANDARD_CFLAGS ?= -DSTANDARD_DATASET

include $(BENCHMARKS_ROOT)/common/carts.mk

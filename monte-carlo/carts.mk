################################################################################
# Monte Carlo defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

MONTE_CARLO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(MONTE_CARLO_ROOT)/..)

# Shared includes for Monte Carlo suite
INCLUDES ?= -I$(MONTE_CARLO_ROOT)/common

include $(BENCHMARKS_ROOT)/common/carts.mk

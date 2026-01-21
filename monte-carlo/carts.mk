################################################################################
# Monte Carlo defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

MONTE_CARLO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(MONTE_CARLO_ROOT)/..)

# Shared includes for Monte Carlo suite
INCLUDES ?= -I$(MONTE_CARLO_ROOT)/common

# Auto-detect arts.cfg at suite level
ifneq ($(wildcard $(MONTE_CARLO_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(MONTE_CARLO_ROOT)/arts.cfg
endif
ifneq ($(wildcard $(BENCHMARKS_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(BENCHMARKS_ROOT)/arts.cfg
endif

include $(BENCHMARKS_ROOT)/common/carts.mk

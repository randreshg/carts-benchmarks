################################################################################
# Graph500 defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

GRAPH500_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(GRAPH500_ROOT)/..)

# Shared includes for Graph500 suite
INCLUDES ?= -I$(GRAPH500_ROOT)/common

# Auto-detect arts.cfg at suite level
ifneq ($(wildcard $(GRAPH500_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(GRAPH500_ROOT)/arts.cfg
endif
ifneq ($(wildcard $(BENCHMARKS_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(BENCHMARKS_ROOT)/arts.cfg
endif

include $(BENCHMARKS_ROOT)/common/carts.mk

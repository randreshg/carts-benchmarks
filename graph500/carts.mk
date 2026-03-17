################################################################################
# Graph500 defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

GRAPH500_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(GRAPH500_ROOT)/..)

# Shared includes for Graph500 suite
INCLUDES ?= -I$(GRAPH500_ROOT)/common

include $(BENCHMARKS_ROOT)/common/carts.mk

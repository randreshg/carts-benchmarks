################################################################################
# SeisSol defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

SEISSOL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(SEISSOL_ROOT)/..)

# Shared includes for SeisSol suite
INCLUDES ?= -I$(SEISSOL_ROOT)/common

include $(BENCHMARKS_ROOT)/common/carts.mk

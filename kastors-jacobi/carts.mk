################################################################################
# KaStORS specific defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

KASTORS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(KASTORS_ROOT)/..)

# Set includes for common/carts.mk
INCLUDES ?= -I. -I../include

include $(BENCHMARKS_ROOT)/common/carts.mk

################################################################################
# SPECFEM3D defaults layered on top of the shared CARTS pipeline.
# Size targets (small/medium/large) are defined in each benchmark's Makefile.
################################################################################

SPECFEM3D_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(SPECFEM3D_ROOT)/..)

# Shared includes for SPECFEM3D suite
INCLUDES ?= -I$(SPECFEM3D_ROOT)/common

include $(BENCHMARKS_ROOT)/common/carts.mk

################################################################################
# KaStORS specific defaults layered on top of the shared CARTS pipeline.
################################################################################

KASTORS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(KASTORS_ROOT)/..)

# Set includes for common/carts.mk
INCLUDES ?= -I. -I../include

# Size configurations for KaStORS benchmarks
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size"
	$(MAKE) all openmp CFLAGS="-DSMALL $(EXTRA_CFLAGS)" LDFLAGS="$(LDFLAGS)" INCLUDES="$(INCLUDES)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size"
	$(MAKE) all openmp CFLAGS="-DMEDIUM $(EXTRA_CFLAGS)" LDFLAGS="$(LDFLAGS)" INCLUDES="$(INCLUDES)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size"
	$(MAKE) all openmp CFLAGS="-DLARGE $(EXTRA_CFLAGS)" LDFLAGS="$(LDFLAGS)" INCLUDES="$(INCLUDES)"

include $(BENCHMARKS_ROOT)/common/carts.mk

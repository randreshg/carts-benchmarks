################################################################################
# ML kernel defaults layered on top of the shared CARTS pipeline.
################################################################################

ML_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(ML_ROOT)/..)

# Load problem sizes if available
PROBLEM_SIZES_FILE := $(ML_ROOT)/common/problem-sizes.mk
ifneq ($(strip $(wildcard $(PROBLEM_SIZES_FILE))),)
include $(PROBLEM_SIZES_FILE)
endif

# Per-benchmark default CFLAGS (from problem-sizes.mk)
DEFAULT_CFLAGS_VAR := DEFAULT_CFLAGS_$(EXAMPLE_NAME)
DEFAULT_EXAMPLE_CFLAGS := $($(DEFAULT_CFLAGS_VAR))
ifneq ($(strip $(DEFAULT_EXAMPLE_CFLAGS)),)
  CFLAGS ?= $(DEFAULT_EXAMPLE_CFLAGS)
endif

# Optional preset targets allow quick sweeps such as `make mini`.
PRESET_TARGETS ?= $(strip $(PRESET_TARGETS_$(EXAMPLE_NAME)))
ifneq ($(strip $(PRESET_TARGETS)),)
define CARTS_PRESET_template
.PHONY: $(1)
$(1):
	$$(MAKE) CFLAGS="$$(PRESET_FLAGS_$(EXAMPLE_NAME)_$(1))" LDFLAGS="$$(LDFLAGS)" INCLUDES="$$(INCLUDES)" all openmp
endef
$(foreach preset,$(PRESET_TARGETS),$(eval $(call CARTS_PRESET_template,$(preset))))
endif

include $(BENCHMARKS_ROOT)/common/carts.mk

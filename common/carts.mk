################################################################################
# Shared CARTS build rules for all benchmarks.
#
# Required variables (set before include):
#   EXAMPLE_NAME - benchmark name (required)
#
# Optional variables:
#   SRC          - source file (default: $(EXAMPLE_NAME).c)
#   INCLUDES     - include flags (e.g., -I. -I../common)
#   LDFLAGS      - linker flags (e.g., -lm)
#   CFLAGS       - additional compiler flags
#   EXTRA_CFLAGS - extra flags appended to CFLAGS
#   ARTS_CFG     - path to arts.cfg (auto-detected if present)
################################################################################

SHELL := $(shell command -v bash 2>/dev/null || echo /bin/bash)
.SHELLFLAGS := -eo pipefail -c

ifndef EXAMPLE_NAME
$(error EXAMPLE_NAME must be defined)
endif

# Defaults
SRC ?= $(EXAMPLE_NAME).c
CARTS ?= carts
BUILD_DIR ?= build
LOG_DIR ?= logs
INCLUDES ?=
LDFLAGS ?=
CFLAGS ?=

# Output files
ARTS_BINARY := $(EXAMPLE_NAME)_arts
OMP_BINARY := $(BUILD_DIR)/$(EXAMPLE_NAME)_omp

# Auto-detect arts.cfg
ARTS_CFG ?= $(firstword $(wildcard arts.cfg))
ifneq ($(strip $(ARTS_CFG)),)
  ARTS_CFG_ARG := --arts-config $(ARTS_CFG)
endif

# Compile flags for carts execute
EXECUTE_FLAGS := --print-debug-info --raise-scf-to-affine -O0 -S $(INCLUDES) $(CFLAGS)

# Compile flags for OpenMP reference
OMP_FLAGS := -fopenmp -O3 $(INCLUDES) $(CFLAGS) -lm -lcartsbenchmarks

.PHONY: all openmp clean

# Build ARTS executable (carts execute -O3 does everything in one step)
all: | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Building ARTS executable"
	@$(CARTS) execute $(if $(LDFLAGS),--compile-args "$(LDFLAGS)") \
		$(SRC) -O3 $(ARTS_CFG_ARG) $(EXECUTE_FLAGS) \
		> $(LOG_DIR)/build.log 2>&1 || (cat $(LOG_DIR)/build.log >&2; exit 1)
	@echo "[$(EXAMPLE_NAME)] Built: $(ARTS_BINARY)"

# Build OpenMP reference executable
$(OMP_BINARY): $(SRC) | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Building OpenMP reference -> $@"
	@$(CARTS) clang $(SRC) $(OMP_FLAGS) $(LDFLAGS) -o $@ \
		2>&1 | tee $(LOG_DIR)/openmp.log; exit $${PIPESTATUS[0]}

openmp: $(OMP_BINARY)

$(BUILD_DIR):
	@mkdir -p $@

$(LOG_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LOG_DIR) $(ARTS_BINARY) *.mlir *.ll .carts-metadata.json *_metadata.mlir

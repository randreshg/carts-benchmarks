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
OMP_CFLAGS_STAMP := $(BUILD_DIR)/.omp_cflags

# Auto-detect arts.cfg
ARTS_CFG ?= $(firstword $(wildcard arts.cfg))
ARTS_CFG_ARG = $(if $(strip $(ARTS_CFG)),--arts-config $(ARTS_CFG),)

# Compile flags for carts execute (cgeist flags like --raise-scf-to-affine, -O0, -S are handled internally)
EXECUTE_FLAGS := $(INCLUDES) $(CFLAGS)
# Extra carts execute flags (e.g., --partition-fallback=fine)
EXECUTE_ARGS ?=

# Compile flags for OpenMP reference
OMP_FLAGS := -fopenmp -O3 $(INCLUDES) $(CFLAGS) -lm -lcartsbenchmarks

.PHONY: all openmp clean

# Build ARTS executable (carts execute -O3 does everything in one step)
all: | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Building ARTS executable"
	@$(CARTS) execute $(if $(LDFLAGS),--compile-args "$(LDFLAGS)") \
		$(EXECUTE_ARGS) $(SRC) -O3 $(ARTS_CFG_ARG) $(EXECUTE_FLAGS) \
		> $(LOG_DIR)/build.log 2>&1 || (cat $(LOG_DIR)/build.log >&2; exit 1)
	@echo "[$(EXAMPLE_NAME)] Built: $(ARTS_BINARY)"

# Track OpenMP build flags to avoid stale binaries when size/CFLAGS change
# FORCE ensures this rule always runs to compare current flags with cached flags
$(OMP_CFLAGS_STAMP): FORCE | $(BUILD_DIR)
	@echo "$(OMP_FLAGS) $(LDFLAGS)" > $@.tmp
	@{ \
	  if [ ! -f "$@" ] || ! cmp -s "$@.tmp" "$@"; then \
	    mv "$@.tmp" "$@"; \
	  else \
	    rm -f "$@.tmp"; \
	  fi; \
	}

# FORCE target ensures rules always execute (standard Make pattern)
FORCE:
.PHONY: FORCE

# Build OpenMP reference executable
$(OMP_BINARY): $(SRC) $(OMP_CFLAGS_STAMP) | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Building OpenMP reference -> $@"
	@$(CARTS) clang $(SRC) $(OMP_FLAGS) $(LDFLAGS) -o $@ \
		2>&1 | tee $(LOG_DIR)/openmp.log; exit $${PIPESTATUS[0]}

openmp: $(OMP_BINARY)

$(BUILD_DIR):
	@mkdir -p $@

$(LOG_DIR):
	@mkdir -p $@

.PHONY: run-arts run-omp

# Run ARTS executable
run-arts: all
	@echo "[$(EXAMPLE_NAME)] Running ARTS..."
	./$(ARTS_BINARY)

# Run OpenMP executable with OMP_WAIT_POLICY=ACTIVE for fair comparison
# (ACTIVE makes idle OMP threads spin-wait, matching ARTS worker behavior)
run-omp: $(OMP_BINARY)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP (OMP_WAIT_POLICY=ACTIVE)..."
	OMP_WAIT_POLICY=ACTIVE ./$(OMP_BINARY)

clean:
	rm -rf $(BUILD_DIR) $(LOG_DIR) $(ARTS_BINARY) *.mlir *.ll .carts-metadata.json *_metadata.mlir

################################################################################
# Size targets - use SMALL_CFLAGS/MEDIUM_CFLAGS/LARGE_CFLAGS/EXTRALARGE_CFLAGS
# from individual Makefile. These variables must be defined BEFORE including this file.
#
# Available targets:
#   small / medium / large / extralarge       - Build both ARTS and OpenMP
#   small-arts / medium-arts / ...            - Build only ARTS executable
#   small-openmp / medium-openmp / ...        - Build only OpenMP executable
#   run-small / run-medium / run-large / ...  - Build and run both variants
################################################################################

.PHONY: small medium large extralarge
.PHONY: small-arts medium-arts large-arts extralarge-arts
.PHONY: small-openmp medium-openmp large-openmp extralarge-openmp
.PHONY: run-small run-medium run-large run-extralarge

# Build both variants with size
small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size"
	$(MAKE) all openmp CFLAGS="$(SMALL_CFLAGS) $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size"
	$(MAKE) all openmp CFLAGS="$(MEDIUM_CFLAGS) $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size"
	$(MAKE) all openmp CFLAGS="$(LARGE_CFLAGS) $(EXTRA_CFLAGS)"

extralarge:
	@echo "[$(EXAMPLE_NAME)] Building with EXTRALARGE size"
	$(MAKE) all openmp CFLAGS="$(EXTRALARGE_CFLAGS) $(EXTRA_CFLAGS)"

# Build only ARTS with size
small-arts:
	@echo "[$(EXAMPLE_NAME)] Building ARTS with SMALL size"
	$(MAKE) all CFLAGS="$(SMALL_CFLAGS) $(EXTRA_CFLAGS)" ARTS_CFG="$(ARTS_CFG)"

medium-arts:
	@echo "[$(EXAMPLE_NAME)] Building ARTS with MEDIUM size"
	$(MAKE) all CFLAGS="$(MEDIUM_CFLAGS) $(EXTRA_CFLAGS)" ARTS_CFG="$(ARTS_CFG)"

large-arts:
	@echo "[$(EXAMPLE_NAME)] Building ARTS with LARGE size"
	$(MAKE) all CFLAGS="$(LARGE_CFLAGS) $(EXTRA_CFLAGS)" ARTS_CFG="$(ARTS_CFG)"

extralarge-arts:
	@echo "[$(EXAMPLE_NAME)] Building ARTS with EXTRALARGE size"
	$(MAKE) all CFLAGS="$(EXTRALARGE_CFLAGS) $(EXTRA_CFLAGS)" ARTS_CFG="$(ARTS_CFG)"

# Build only OpenMP with size
small-openmp:
	@echo "[$(EXAMPLE_NAME)] Building OpenMP with SMALL size"
	$(MAKE) openmp CFLAGS="$(SMALL_CFLAGS) $(EXTRA_CFLAGS)"

medium-openmp:
	@echo "[$(EXAMPLE_NAME)] Building OpenMP with MEDIUM size"
	$(MAKE) openmp CFLAGS="$(MEDIUM_CFLAGS) $(EXTRA_CFLAGS)"

large-openmp:
	@echo "[$(EXAMPLE_NAME)] Building OpenMP with LARGE size"
	$(MAKE) openmp CFLAGS="$(LARGE_CFLAGS) $(EXTRA_CFLAGS)"

extralarge-openmp:
	@echo "[$(EXAMPLE_NAME)] Building OpenMP with EXTRALARGE size"
	$(MAKE) openmp CFLAGS="$(EXTRALARGE_CFLAGS) $(EXTRA_CFLAGS)"

# Build and run both variants with size
run-small: small
	@echo "[$(EXAMPLE_NAME)] Running ARTS (SMALL)..."
	./$(ARTS_BINARY)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP (SMALL, OMP_WAIT_POLICY=ACTIVE)..."
	OMP_WAIT_POLICY=ACTIVE ./$(OMP_BINARY)

run-medium: medium
	@echo "[$(EXAMPLE_NAME)] Running ARTS (MEDIUM)..."
	./$(ARTS_BINARY)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP (MEDIUM, OMP_WAIT_POLICY=ACTIVE)..."
	OMP_WAIT_POLICY=ACTIVE ./$(OMP_BINARY)

run-large: large
	@echo "[$(EXAMPLE_NAME)] Running ARTS (LARGE)..."
	./$(ARTS_BINARY)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP (LARGE, OMP_WAIT_POLICY=ACTIVE)..."
	OMP_WAIT_POLICY=ACTIVE ./$(OMP_BINARY)

run-extralarge: extralarge
	@echo "[$(EXAMPLE_NAME)] Running ARTS (EXTRALARGE)..."
	./$(ARTS_BINARY)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP (EXTRALARGE, OMP_WAIT_POLICY=ACTIVE)..."
	OMP_WAIT_POLICY=ACTIVE ./$(OMP_BINARY)

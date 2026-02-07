EXAMPLE_NAME := lulesh
SRC := lulesh.c

# Size configurations - use -s flag at runtime
# TINY: 3x3x3 (27 elements), SMALL: 10x10x10, MEDIUM: 30x30x30
# LARGE (64-core): 80^3=512000=64x8000 elements, 500 iters
# EXTRALARGE (1920-worker): 120^3=1728000=1920x900 elements, 750 iters. 5.06x LARGE
TINY_CFLAGS ?=
SMALL_CFLAGS ?=
MEDIUM_CFLAGS ?=
LARGE_CFLAGS ?=
EXTRALARGE_CFLAGS ?=
EXTRA_CFLAGS ?= -DCARTS_QSTOP=1.0e+30
LDFLAGS ?= -lm -ldl

# Runtime arguments for different sizes
TINY_ARGS := -s 3 -i 5
SMALL_ARGS := -s 10 -i 10
MEDIUM_ARGS := -s 30 -i 500
LARGE_ARGS := -s 80 -i 500
EXTRALARGE_ARGS := -s 120 -i 750

include ../common/carts.mk

# Override run targets to pass size arguments
run-tiny: tiny
	@echo "[$(EXAMPLE_NAME)] Running ARTS with tiny size (3^3)"
	./$(ARTS_BINARY) $(TINY_ARGS)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP with tiny size (3^3)"
	./$(OMP_BINARY) $(TINY_ARGS)

run-small: small
	@echo "[$(EXAMPLE_NAME)] Running ARTS with small size (10^3)"
	./$(ARTS_BINARY) $(SMALL_ARGS)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP with small size (10^3)"
	./$(OMP_BINARY) $(SMALL_ARGS)

run-medium: medium
	@echo "[$(EXAMPLE_NAME)] Running ARTS with medium size (30^3)"
	./$(ARTS_BINARY) $(MEDIUM_ARGS)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP with medium size (30^3)"
	./$(OMP_BINARY) $(MEDIUM_ARGS)

run-large: large
	@echo "[$(EXAMPLE_NAME)] Running ARTS with large size (80^3)"
	./$(ARTS_BINARY) $(LARGE_ARGS)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP with large size (80^3)"
	./$(OMP_BINARY) $(LARGE_ARGS)

run-extralarge: extralarge
	@echo "[$(EXAMPLE_NAME)] Running ARTS with extralarge size (120^3)"
	./$(ARTS_BINARY) $(EXTRALARGE_ARGS)
	@echo "[$(EXAMPLE_NAME)] Running OpenMP with extralarge size (120^3)"
	./$(OMP_BINARY) $(EXTRALARGE_ARGS)

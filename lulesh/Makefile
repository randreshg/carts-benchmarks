EXAMPLE_NAME := lulesh
SRC := lulesh.c

# Size configurations - use -s flag at runtime
# TINY: 3x3x3 (27 elements), SMALL: 10x10x10, MEDIUM: 30x30x30
TINY_CFLAGS ?=
SMALL_CFLAGS ?=
MEDIUM_CFLAGS ?=
EXTRA_CFLAGS ?= -DCARTS_QSTOP=1.0e+30
LDFLAGS ?= -lm -ldl

# Runtime arguments for different sizes
TINY_ARGS := -s 3 -i 5
SMALL_ARGS := -s 10 -i 10
MEDIUM_ARGS := -s 30 -i 500
LARGE_ARGS := -s 50 -i 1000

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

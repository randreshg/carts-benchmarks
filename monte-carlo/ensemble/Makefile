EXAMPLE_NAME := monte_carlo_ensemble
SRC := monte_carlo_ensemble.c
LDFLAGS := -lm -ldl

# Size configurations for Monte Carlo Ensemble
# Peak memory = num_threads * STATE_DIM^2 * 8 bytes (alloc inside loop)
# LARGE (64-core): 12800=64x200 samples, peak ~256MB
# EXTRALARGE (1920-worker): 61440=1920x32 samples, peak ~240MB/node. 4.8x LARGE
SMALL_CFLAGS ?= -DNUM_SAMPLES=1000 -DSTATE_DIM=1024
MEDIUM_CFLAGS ?= -DNUM_SAMPLES=5000 -DSTATE_DIM=1024
LARGE_CFLAGS ?= -DNUM_SAMPLES=12800 -DSTATE_DIM=1024
EXTRALARGE_CFLAGS ?= -DNUM_SAMPLES=61440 -DSTATE_DIM=1024

include ../carts.mk

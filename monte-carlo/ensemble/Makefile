EXAMPLE_NAME := monte_carlo_ensemble
SRC := monte_carlo_ensemble.c
LDFLAGS := -lm -ldl

# Size configurations for Monte Carlo Ensemble
# Peak memory = num_threads * STATE_DIM^2 * 8 bytes (alloc inside loop)
# SMALL: 1000 samples, 128x128 state
# MEDIUM: 2000 samples, 256x256 state
# LARGE (64-core): 12800=64x200 samples, 512x512 state
# EXTRALARGE (1920-worker): 61440=1920x32 samples, 512x512 state
SMALL_CFLAGS ?= -DNUM_SAMPLES=1000 -DSTATE_DIM=128
MEDIUM_CFLAGS ?= -DNUM_SAMPLES=2000 -DSTATE_DIM=256
LARGE_CFLAGS ?= -DNUM_SAMPLES=12800 -DSTATE_DIM=512
EXTRALARGE_CFLAGS ?= -DNUM_SAMPLES=61440 -DSTATE_DIM=512

include ../carts.mk

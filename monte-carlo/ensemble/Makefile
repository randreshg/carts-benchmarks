EXAMPLE_NAME := monte_carlo_ensemble
SRC := monte_carlo_ensemble.c
LDFLAGS := -lm

# Size configurations for Monte Carlo Ensemble
# Memory = NUM_SAMPLES * STATE_DIM^2 * 8 bytes
# SMALL: 1000 samples x 1024^2 = ~8 GB
# MEDIUM: 10000 samples x 1024^2 = ~80 GB (exceeds single node)
# LARGE: 50000 samples x 1024^2 = ~400 GB
# EXTRALARGE: 100000 samples x 1024^2 = ~800 GB
SMALL_CFLAGS ?= -DNUM_SAMPLES=1000 -DSTATE_DIM=1024
MEDIUM_CFLAGS ?= -DNUM_SAMPLES=10000 -DSTATE_DIM=1024
LARGE_CFLAGS ?= -DNUM_SAMPLES=50000 -DSTATE_DIM=1024
EXTRALARGE_CFLAGS ?= -DNUM_SAMPLES=100000 -DSTATE_DIM=1024

# Link math library for sin/cos/exp functions
LDFLAGS := -lm

include ../carts.mk

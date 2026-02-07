EXAMPLE_NAME := volume_integral
SRC := volume_integral.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 24000000=64x375000. EXTRALARGE (1920-worker): 120000000=1920x62500. 5.0x LARGE
SMALL_CFLAGS ?= -DN_ELEMENTS=10000
MEDIUM_CFLAGS ?= -DN_ELEMENTS=100000
LARGE_CFLAGS ?= -DN_ELEMENTS=24000000
EXTRALARGE_CFLAGS ?= -DN_ELEMENTS=120000000

include ../carts.mk

EXAMPLE_NAME := volume_integral
SRC := volume_integral.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 24000000=64x375000. EXTRALARGE (1920-worker): 96000000=1920x50000. 4.0x LARGE
SMALL_CFLAGS ?= -DN_ELEMENTS=10000
MEDIUM_CFLAGS ?= -DN_ELEMENTS=100000
LARGE_CFLAGS ?= -DN_ELEMENTS=2304000 -DNREPS=10
EXTRALARGE_CFLAGS ?= -DN_ELEMENTS=9216000 -DNREPS=10

include ../carts.mk

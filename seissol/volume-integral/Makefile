EXAMPLE_NAME := volume_integral
SRC := volume_integral.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 40000000=64x625000. EXTRALARGE (1920-worker): 192000000=1920x100000. 4.8x LARGE
SMALL_CFLAGS ?= -DN_ELEMENTS=10000
MEDIUM_CFLAGS ?= -DN_ELEMENTS=100000
LARGE_CFLAGS ?= -DN_ELEMENTS=40000000
EXTRALARGE_CFLAGS ?= -DN_ELEMENTS=192000000

include ../carts.mk

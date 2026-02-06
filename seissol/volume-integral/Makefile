EXAMPLE_NAME := volume_integral
SRC := volume_integral.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# SMALL: 10000 elements, MEDIUM: 100000, LARGE: 500000, EXTRALARGE: 2000000
SMALL_CFLAGS ?= -DN_ELEMENTS=10000
MEDIUM_CFLAGS ?= -DN_ELEMENTS=100000
LARGE_CFLAGS ?= -DN_ELEMENTS=500000
EXTRALARGE_CFLAGS ?= -DN_ELEMENTS=40000000

include ../carts.mk

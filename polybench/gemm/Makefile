EXAMPLE_NAME := gemm
SRC := gemm.c
LDFLAGS := -lm -ldl

# Size configurations - can be overridden via command line
# LARGE (64-core): 5120 (80x64). EXTRALARGE (1920-worker): 9600 (5x1920). 6.6x LARGE
SMALL_CFLAGS ?= -DNI=128 -DNJ=128 -DNK=128
MEDIUM_CFLAGS ?= -DNI=1024 -DNJ=1024 -DNK=1024
LARGE_CFLAGS ?= -DNI=5120 -DNJ=5120 -DNK=5120
EXTRALARGE_CFLAGS ?= -DNI=9600 -DNJ=9600 -DNK=9600

include ../carts.mk

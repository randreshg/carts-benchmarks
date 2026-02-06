EXAMPLE_NAME := velocity_update
SRC := velocity_update.c

# Size configurations - can be overridden via command line
# SMALL: 20x20x20 (~8K), MEDIUM: 64x64x64 (~262K), LARGE: 128x128x128 (~2M), EXTRALARGE: 256x256x256 (~16M)
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=64 -DNY=64 -DNZ=64
LARGE_CFLAGS ?= -DNX=128 -DNY=128 -DNZ=128
EXTRALARGE_CFLAGS ?= -DNX=500 -DNY=500 -DNZ=500

include ../carts.mk

EXAMPLE_NAME := vel4sg_base
SRC := vel4sg_base.c

# Size configurations - can be overridden via command line
# SMALL: 20x20x20 (~8K), MEDIUM: 40x40x40 (~64K), LARGE: 60x60x60 (~216K), EXTRALARGE: 100x100x100 (~1M)
# Note: Larger sizes cause FP precision differences between ARTS/OMP compilers
SMALL_CFLAGS ?= -DNX=20 -DNY=20 -DNZ=20
MEDIUM_CFLAGS ?= -DNX=40 -DNY=40 -DNZ=40
LARGE_CFLAGS ?= -DNX=60 -DNY=60 -DNZ=60
EXTRALARGE_CFLAGS ?= -DNX=512 -DNY=512 -DNZ=512

include ../carts.mk

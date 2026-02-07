EXAMPLE_NAME := stream
SRC := stream.c
LDFLAGS := -lm -ldl

# Size configurations (array elements)
# Memory per array = elements * 8 bytes (double)
# Total memory = 3 arrays * per-array size
# EXTRALARGE: 2.4B = 1920 x 1250000 (~57.6GB total). 4.8x LARGE
SMALL_CFLAGS ?= -DSTREAM_ARRAY_SIZE=1000000 -DNTIMES=10      # ~24MB total
MEDIUM_CFLAGS ?= -DSTREAM_ARRAY_SIZE=10000000 -DNTIMES=10    # ~240MB total (default)
LARGE_CFLAGS ?= -DSTREAM_ARRAY_SIZE=500000000 -DNTIMES=10    # ~12GB total
EXTRALARGE_CFLAGS ?= -DSTREAM_ARRAY_SIZE=2400000000 -DNTIMES=10 # ~57.6GB total

include ../common/carts.mk

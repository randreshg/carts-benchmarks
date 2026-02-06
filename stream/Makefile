EXAMPLE_NAME := stream
SRC := stream.c
LDFLAGS := -lm -ldl

# Size configurations (array elements)
# Memory per array = elements * 8 bytes (double)
# Total memory = 3 arrays * per-array size
SMALL_CFLAGS ?= -DSTREAM_ARRAY_SIZE=1000000 -DNTIMES=10      # ~24MB total
MEDIUM_CFLAGS ?= -DSTREAM_ARRAY_SIZE=10000000 -DNTIMES=10    # ~240MB total (default)
LARGE_CFLAGS ?= -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10    # ~2.4GB total
EXTRALARGE_CFLAGS ?= -DSTREAM_ARRAY_SIZE=500000000 -DNTIMES=10 # ~19.2GB total

include ../common/carts.mk

EXAMPLE_NAME := stream
SRC := stream.c
LDFLAGS := -lm -ldl

# Size configurations (array elements)
# Memory per array = elements * 8 bytes (double)
# Total memory = 3 arrays * per-array size
# LARGE: 700M elements (~17GB total). EXTRALARGE: 2.8B elements (~67GB total)
SMALL_CFLAGS ?= -DSTREAM_ARRAY_SIZE=1000000 -DNTIMES=10      # ~24MB total
MEDIUM_CFLAGS ?= -DSTREAM_ARRAY_SIZE=10000000 -DNTIMES=10    # ~240MB total (default)
LARGE_CFLAGS ?= -DSTREAM_ARRAY_SIZE=700000000 -DNTIMES=10    # ~17GB total
EXTRALARGE_CFLAGS ?= -DSTREAM_ARRAY_SIZE=2800000000 -DNTIMES=10 # ~67GB total

include ../common/carts.mk

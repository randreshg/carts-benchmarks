################################################################################
# Default problem sizes for ML kernels.
# Size targets (small/medium/large/extralarge) are defined in each Makefile.
################################################################################

# Default configurations (used when building without size target)
DEFAULT_CFLAGS_activations := -DSIZE=1048576
DEFAULT_CFLAGS_batchnorm := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32
DEFAULT_CFLAGS_pooling := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=64 -DWIDTH=64 -DPOOL_SIZE=2
DEFAULT_CFLAGS_layernorm := -DBATCH=16 -DHIDDEN=1024

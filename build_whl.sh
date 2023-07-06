#!/bin/bash
# For now have to build the deepspeed wheel in the ngc pytorch docker container which has pytorch 2.0 compiled with cuda 12.1.  
# Pytorch compiled with cuda 12.1 is not provided by torch team yet.
#
# Build all ops other than sparse attn which requires a pinned version of triton.
# Build for same arch list as that of the ngc pytorch container excluding any below pascal which do not have fp16 support.
# 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90'
TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 9.0" DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 python setup.py build_ext bdist_wheel

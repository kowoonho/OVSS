#!/usr/bin/env bash

# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=1,2,3,4
# export OMP_NUM_THREADS=2

SCRIPT=$1
CONFIG=$2
GPUS=$3
PORT=${PORT:-29800}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $SCRIPT --cfg $CONFIG ${@:4}

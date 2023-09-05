#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

if [ ! -d "work_dirs" ]; then
  mkdir work_dirs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} > work_dirs/train.log 2>&1 &

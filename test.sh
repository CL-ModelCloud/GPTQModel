#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

start_time=$(date +%s)

pytest --durations=0 tests/models/test_qwen2_5.py

execution_time=$(( $(date +%s) - start_time ))

echo "$((execution_time / 60))m $((execution_time % 60))s"

curl "http://10.0.13.31/gpu/vram?gpu=$CUDA_VISIBLE_DEVICES&range=$execution_time&unit=second"


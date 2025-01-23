#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")/..";pwd)
export PYTHONPATH=${WORK_DIR}
CHCEKPOINT_PATH=$1
LM_MODEL_TYPE=$2
OUTPUT_DIR_EVAL=$(cd "$(dirname "${CHCEKPOINT_PATH}")";pwd)/mobilevlm-3.evaluation
mkdir -p ${OUTPUT_DIR_EVAL}
CONV_MODE=v1
cd ${WORK_DIR}

DATASET_NAME=gqa
MODEL_GENERATOR=eve.eval.model_vqa_loader
DATA_ROOT=/cache/data/VLM/eve/benchmark_data/gqa
SPLIT_NAME=llava_gqa_testdev_balanced
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}


#!/bin/bash

GPUS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPUS"
CHUNKS=${#GPULIST[@]}

MODEL_LOADER=$1
MODEL_DIR=$2
CONV_MODE=$3
SPLIT_NME=$4
DATA_ROOT=$5
SAVE_PATH=$6/${SPLIT_NME}
LM_MODEL_TYPE=$7
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ${MODEL_LOADER} \
        --model-path ${MODEL_DIR} \
        --lm_model_type ${LM_MODEL_TYPE} \
        --question-file ${DATA_ROOT}/${SPLIT_NME}.tsv \
        --answers-file ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --lang en \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode ${CONV_MODE} &
done

wait

RESULT_FILE=${SAVE_PATH}/predictions.jsonl
> "${RESULT_FILE}"  # Clear out the output file if it exists
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl >> "${RESULT_FILE}"  # Loop through the indices and concatenate each file
    rm ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl
done

python ${DATA_ROOT}/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_ROOT}/${SPLIT_NME}.tsv \
    --result-dir ${SAVE_PATH} \
    --upload-dir ${SAVE_PATH} \
    --experiment predictions

python ${DATA_ROOT}/eval.py --result ${SAVE_PATH}/predictions.xlsx --meta ${DATA_ROOT}/${SPLIT_NME}.tsv

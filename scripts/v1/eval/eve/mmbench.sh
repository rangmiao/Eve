#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_DIR=$1
CONV_MODE=$2
SPLIT_NME=$3
DATA_ROOT=$4
SAVE_PATH=$5/${SPLIT_NME}


for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) eve/eval/model_vqa_mmbench.py \
        --model-path ${MODEL_DIR} \
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

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
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) eve/eval/model_vqa_loader.py \
        --model-path ${MODEL_DIR} \
        --question-file ${DATA_ROOT}/${SPLIT_NME}.jsonl \
        --image-folder ${DATA_ROOT}/images \
        --answers-file ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV_MODE} &
done

wait

RESULT_FILE=${SAVE_PATH}/merge.jsonl
> "${RESULT_FILE}"  # Clear out the output file if it exists
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl >> "${RESULT_FILE}"  # Loop through the indices and concatenate each file
    rm ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl
done

echo $RESULT_FILE
python ${DATA_ROOT}/convert_gqa_for_eval.py --src ${RESULT_FILE} --dst ${SAVE_PATH}/merge.json

python ${DATA_ROOT}/eval.py \
    --questions ${DATA_ROOT}/testdev_balanced_questions.json \
    --predictions ${SAVE_PATH}/merge.json

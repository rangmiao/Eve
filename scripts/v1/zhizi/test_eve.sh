
CHCEKPOINT_PATH=$1 # ${OUTPUT_DIR}'/llavaZhizi-1.5b-4.finetune-moe'


OUTPUT_DIR_EVAL=$2
DATASET_NAME=gqa
CONV_MODE=v1
DATA_ROOT=benchmark_data/gqa
SPLIT_NAME=llava_gqa_testdev_balanced
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/v1/eval/eve/${DATASET_NAME}.sh \
    ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=sqa
DATA_ROOT=benchmark_data/sqa
SPLIT_NAME=llava_test_CQM-A
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/v1/eval/eve/${DATASET_NAME}.sh \
    ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}


DATASET_NAME=textvqa
DATA_ROOT=benchmark_data/textvqa
SPLIT_NAME=llava_textvqa_val_v051_ocr
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/eve/${DATASET_NAME}.sh \
    ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=pope
DATA_ROOT=benchmark_data/pope
SPLIT_NAME=llava_pope_test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/eve/${DATASET_NAME}.sh \
    ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}


DATASET_NAME=mme
DATA_ROOT=benchmark_data/mme
SPLIT_NAME=llava_mme
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/eve/${DATASET_NAME}.sh \
    ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=mmbench
DATA_ROOT=benchmark_data/mmbench
SPLIT_NAME=mmbench_dev_en_20231003
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/eve/${DATASET_NAME}.sh \
    ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

# Task Configs
IMAGE_SIZE="4096"
TASK=gs_classification

# Experiment Configs
TOTAL_FOLDS=5 
EXPERIMENT_ID="" 
EPOCHS=30 
BATCH_SIZE=8  
LR=5e-5 
UPDATE_FREQ=3 
WARMUP_EPOCHS=5

LOCAL_GPU_ID=0

# Paths
IMAGE_DIR=""
FINETUNE_PATH=""
DATA_PATH=""
OUTPUT_BASE="${TASK}_${IMAGE_SIZE}/${EXPERIMENT_ID}/"


for (( K_FOLD=4; K_FOLD<5; K_FOLD++ )); do

    LOG_DIR=${OUTPUT_BASE}${K_FOLD}"/log"
    OUTPUT_DIR=${OUTPUT_BASE}${K_FOLD}

    echo "Running fold $K_FOLD"
    
    CUDA_VISIBLE_DEVICES="${LOCAL_GPU_ID}" python run_longvit_finetuning.py \
        --input_size ${IMAGE_SIZE} \
        --model longvit_small_patch32_${IMAGE_SIZE} \
        --task ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --layer_decay 1.0 \
        --lr ${LR} \
        --update_freq ${UPDATE_FREQ} \
        --epochs ${EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --drop_path 0.1 \
        --finetune ${FINETUNE_PATH} \
        --data_path ${DATA_PATH} \
        --image_dir ${IMAGE_DIR}  \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${LOG_DIR} \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --k_fold ${K_FOLD} \
        --num_workers 3 \
        --model_key teacher \
        --randaug \
        --target_norm_mean disable \
        --target_norm_std disable
done
#!/bin/bash

DATA_DIR="./data" 
LOG_DIR_ROOT="./logs"
TOTAL_SUBSET_SIZE=1000

CAE_FEATURE_CHANNELS=128
CAE_LATENT_CHANNELS=64
CAE_RES_BLOCKS=6
CM_FEATURE_CHANNELS=32
Q_CENTERS=16

EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-3
LR_DECAY_STEP=50
LR_DECAY_GAMMA=0.1
WEIGHT_DECAY=1e-5

BETA=0.1
H_TARGETS=(0.5 0.4 0.3 0.2 0.1)

for H_TARGET in "${H_TARGETS[@]}"
do
    python3 src/train.py \
        --data_dir $DATA_DIR \
        --log_dir_root $LOG_DIR_ROOT \
        --total_subset_size $TOTAL_SUBSET_SIZE \
        --cae_feature_channels $CAE_FEATURE_CHANNELS \
        --cae_latent_channels $CAE_LATENT_CHANNELS \
        --cae_res_blocks $CAE_RES_BLOCKS \
        --cm_feature_channels $CM_FEATURE_CHANNELS \
        --q_centers $Q_CENTERS \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --lr_decay_step $LR_DECAY_STEP \
        --lr_decay_gamma $LR_DECAY_GAMMA \
        --weight_decay $WEIGHT_DECAY \
        --beta $BETA \
        --h_target $H_TARGET
done

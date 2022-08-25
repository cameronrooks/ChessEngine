#!/bin/bash

#model hyperparameters and directory name
BATCH_SIZE=512
EPOCHS=150
MODEL_NAME="model2"

#DO NOT EDIT THESE
MODEL_DIR="./trained_models/$MODEL_NAME"
WEIGHT_DIR="$MODEL_DIR/epochs"
PLOT_DIR="$MODEL_DIR/plots"

mkdir -p $WEIGHT_DIR
mkdir -p $PLOT_DIR

python train.py $BATCH_SIZE $EPOCHS $MODEL_DIR

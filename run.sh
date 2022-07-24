#!/bin/bash

BATCH_SIZE=256
EPOCHS=150
MODEL_DIR="./trained_models/model10"

python train.py $BATCH_SIZE $EPOCHS $MODEL_DIR

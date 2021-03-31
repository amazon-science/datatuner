#!/bin/bash
source ./config.sh
TRAINING_DATA_FOLDER=$1
OUTPUT_FOLDER=$2
TRAINING_ARGS=$3
NUM_PARALLEL=$4

if [ -z "$NUM_PARALLEL" ]; then
    NUM_PARALLEL=1
fi

mkdir -p $OUTPUT_FOLDER

echo "Training the classifier and writing the trained model to $OUTPUT_FOLDER"

$python -m torch.distributed.launch --nproc_per_node=$NUM_PARALLEL ../src/datatuner/classification/run_classifier.py  \
--data_dir $TRAINING_DATA_FOLDER  \
--output_dir  $OUTPUT_FOLDER  \
--retrain_base $TRAINING_ARGS
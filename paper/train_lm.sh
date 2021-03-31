#!/bin/bash
source ./config.sh

DATASET=$1
SYSTEM=$2
OUTPUT_FOLDER=$3
NUM_PARALLEL=$4

if [ -z "$NUM_PARALLEL" ]; then
    NUM_PARALLEL=1
fi

SUFFIX=""
if [[ "$SYSTEM" = "DataTuner_No_FC_No_FS" ]]; then
    SUFFIX="_cg"
fi

echo "Training the model for the dataset $DATASET and writing the trained model to $OUTPUT_FOLDER"

$python -m torch.distributed.launch  --nproc_per_node=$NUM_PARALLEL ../src/datatuner/lm/train.py  \
--retrain_base ./lm_training_args/$DATASET/${SYSTEM}_model_training_args.json  \
--logdir $OUTPUT_FOLDER  \
--dataset_path ../data/$DATASET \
--task_config ./task_configs/${DATASET}${SUFFIX}.json \
--ignore_cache \
--overwrite_output_dir
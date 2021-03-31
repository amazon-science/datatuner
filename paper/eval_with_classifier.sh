#!/bin/bash
# example: bash eval_with_classifier.sh ./data/consistency/viggo eval_results ~/trained_classifiers/viggo/ amrl text
source ./config.sh

TRAINING_DATA_FOLDER=$1
GENERATED_DATA_FOLDER=$2
MODEL_FOLDER=$3
DATA_KEY=$4
TEXT_KEY=$5

cp $TRAINING_DATA_FOLDER/labels.txt $GENERATED_DATA_FOLDER/labels.txt

$python ../src/datatuner/classification/classify_generated.py generate \
--in_file $GENERATED_DATA_FOLDER/generated.json \
--model_folder $MODEL_FOLDER \
--data_key $DATA_KEY \
--text_key $TEXT_KEY \̄
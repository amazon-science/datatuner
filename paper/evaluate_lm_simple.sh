#!/bin/bash
source ./config.sh

TEST_FILE=$1
MODEL=$2

echo "Evaluating $TEST_FILE with the model in $MODEL"

$python ../src/datatuner/lm/evaluate.py  \
--filename $TEST_FILE \
--no_sample \
--model_checkpoint $MODEL \
--nbest 5 \
--beam_width 5 \
--per_step_predictions 5 \
--averaging default \
--beam_alpha 0.75 \
--model_type gpt2
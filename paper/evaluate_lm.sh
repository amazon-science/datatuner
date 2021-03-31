#!/bin/bash
source ./config.sh

TEST_FILE=$1
MODEL=$2
NUM_GPUS=$3
PER_GPU=$4
MAX_DATA=$5
SPLITS=$((NUM_GPUS * PER_GPU))
echo "SPLITS: " $SPLITS

if [ -z "$MAX_DATA" ]; then
    MAX_DATA=0
fi

echo "MAX_DATA": $MAX_DATA

CHUNKED_DATA_FOLDER=$(mktemp -d)
echo "Chunked data outputted to the folder $CHUNKED_DATA_FOLDER"

CODE_DIR=../src/datatuner/lm/
# Split data into chunks
$python $CODE_DIR/process_json.py split $TEST_FILE $CHUNKED_DATA_FOLDER $SPLITS

COMMON_ARGUMENTS="--model_checkpoint $MODEL  \
    --no_sample \
    --beam_width 5 \
    --nbest 5 \
    --per_step_predictions 5 \
    --model_type gpt2"

pids=
MAX_SPLITS=$((SPLITS - 1))
RESULTS_FOLDER=$MODEL/$(date +'%Y-%m-%d_%H-%M-%S')
mkdir -p $RESULTS_FOLDER

# Evaluate each chunk
for ((i=0; i<=MAX_SPLITS; i++)); do
    echo "Chunk $i"
    CUDA_VISIBLE_DEVICES=$(($i % $NUM_GPUS)) $python $CODE_DIR/evaluate.py \
    --filename $CHUNKED_DATA_FOLDER/chunk_$i.json \
    --out_folder ${RESULTS_FOLDER}/chunks/chunk_$i \
    --max_data $MAX_DATA \
    ""$COMMON_ARGUMENTS"" &
    pids+=" $!"
done
wait $pids || { echo "there were errors" >&2; rm -rf ${RESULTS_FOLDER}; exit 1; }

# Combine results from all chunks
$python $CODE_DIR/process_json.py combine $RESULTS_FOLDER $SPLITS
GLOBAL_MAX_DATA=$((SPLITS * MAX_DATA))
echo "GLOBAL_MAX_DATA": $GLOBAL_MAX_DATA
CUDA_VISIBLE_DEVICES=0
$python $CODE_DIR/evaluate.py  \
--filename  $TEST_FILE \
--out_folder ${RESULTS_FOLDER} \
--max_data $GLOBAL_MAX_DATA \
""$COMMON_ARGUMENTS""

echo "removing intermediary results from ${RESULTS_FOLDER}/chunks"
rm -rf ${RESULTS_FOLDER}/"chunks"

echo "Final results available in" $RESULTS_FOLDER
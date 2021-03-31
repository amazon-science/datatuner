echo "reading configuration variables"

# Folders of data that cannot be automatically downloaded
# Change the two directories below to the correct ones in your case
# Download it from https://catalog.ldc.upenn.edu/LDC2017T10"
LDC2017_DATA_LOCATION=~/Downloads/abstract_meaning_representation_amr_2.0
# Download it from https://nlds.soe.ucsc.edu/viggo"
VIGGO_DATA_LOCATION=~/Downloads/viggo-v1/

python=~/miniconda3/envs/finetune/bin/python

LM_MODELS_DIR=~/trained_lms
CLASSIFIER_MODELS_DIR=~/trained_classifiers
REPO_FOLDER=DataTuner
TMP_DATA_FOLDER=./tmp
DATA_FOLDER=../data

PAPER_FOLDER_PATTERN=$REPO_FOLDER/paper

# Check if you're running in the correct folder
assert_run_dir() {
    # params: current_dir
    if [[ "$PWD" != *$1 ]]; then
        echo "You should run this script from the folder '$1'. Exiting"
        exit
    fi
}

newline() {
    printf "\n"
}

assert_run_dir $PAPER_FOLDER_PATTERN


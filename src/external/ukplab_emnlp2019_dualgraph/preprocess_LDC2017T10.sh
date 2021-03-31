#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./preprocess_LDC2017T10.sh <dataset_folder> <embeddings_file>"
  exit 2
fi

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bash ${ROOT_DIR}/process_amr/gen_LDC2017T10.sh ${1}

python ${ROOT_DIR}/process_amr/generate_input_opennmt.py -i ${ROOT_DIR}/process_amr/data/amr_ldc2017t10/

mkdir -p ${ROOT_DIR}/data/ldc2017t10
mv ${ROOT_DIR}/process_amr/data/amr_ldc2017t10/dev-* ${ROOT_DIR}/data/ldc2017t10
mv ${ROOT_DIR}/process_amr/data/amr_ldc2017t10/test-* ${ROOT_DIR}/data/ldc2017t10
mv ${ROOT_DIR}/process_amr/data/amr_ldc2017t10/train-* ${ROOT_DIR}/data/ldc2017t10

rm -rf data/ldc2017t10.*
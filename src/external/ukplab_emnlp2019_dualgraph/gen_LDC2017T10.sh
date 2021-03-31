#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p ${ROOT_DIR}/data
REPO_DIR=${ROOT_DIR}/data/

DATA_DIR=${1}
mkdir -p ${REPO_DIR}/tmp_amr
PREPROC_DIR=${REPO_DIR}/tmp_amr
ORIG_AMR_DIR=${DATA_DIR}/data/amrs/split
mkdir -p ${REPO_DIR}/amr_ldc2017t10
FINAL_AMR_DIR=${REPO_DIR}/amr_ldc2017t10


mkdir -p ${PREPROC_DIR}/train
mkdir -p ${PREPROC_DIR}/dev
mkdir -p ${PREPROC_DIR}/test

mkdir -p ${FINAL_AMR_DIR}/train
mkdir -p ${FINAL_AMR_DIR}/dev
mkdir -p ${FINAL_AMR_DIR}/test


cat ${ORIG_AMR_DIR}/training/amr-* > ${PREPROC_DIR}/train/raw_amrs.txt
cat ${ORIG_AMR_DIR}/dev/amr-* > ${PREPROC_DIR}/dev/raw_amrs.txt
# cat ${ORIG_AMR_DIR}/test/amr-* > ${PREPROC_DIR}/test/*_raw_amrs.txt
# cat ${ORIG_AMR_DIR}/test/1_amr-release-2.0-alignments-test-proxy.txt ${ORIG_AMR_DIR}/test/2_amr-release-2.0-alignments-test-dfa.txt ${ORIG_AMR_DIR}/test/3_amr-release-2.0-alignments-test-bolt.txt ${ORIG_AMR_DIR}/test/4_amr-release-2.0-alignments-test-consensus.txt ${ORIG_AMR_DIR}/test/5_amr-release-2.0-alignments-test-xinhua.txt > ${PREPROC_DIR}/test/raw_amrs.txt
cat ${ORIG_AMR_DIR}/test/1_amr-release-2.0-amrs-test-proxy.txt ${ORIG_AMR_DIR}/test/2_amr-release-2.0-amrs-test-dfa.txt ${ORIG_AMR_DIR}/test/3_amr-release-2.0-amrs-test-bolt.txt ${ORIG_AMR_DIR}/test/4_amr-release-2.0-amrs-test-xinhua.txt  ${ORIG_AMR_DIR}/test/5_amr-release-2.0-amrs-test-consensus.txt> ${PREPROC_DIR}/test/raw_amrs.txt


for SPLIT in test dev train ; do
    echo "processing $SPLIT..."
    # get the surface and the graphs separately
    python ${ROOT_DIR}/split_amr.py ${PREPROC_DIR}/${SPLIT}/raw_amrs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${PREPROC_DIR}/${SPLIT}/graphs.txt
    
    python ${ROOT_DIR}/preproc_amr.py ${PREPROC_DIR}/${SPLIT}/graphs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${FINAL_AMR_DIR}/${SPLIT}/nodes.pp.txt ${FINAL_AMR_DIR}/${SPLIT}/surface.pp.txt --mode LIN --triples-output  ${FINAL_AMR_DIR}/${SPLIT}/triples.pp.txt
    # python ${ROOT_DIR}/preproc_amr.py ${PREPROC_DIR}/${SPLIT}/graphs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${FINAL_AMR_DIR}/${SPLIT}/nodes.pp.txt ${FINAL_AMR_DIR}/${SPLIT}/surface.pp.txt --mode LINE_GRAPH --triples-output ${FINAL_AMR_DIR}/${SPLIT}/triples.pp.txt
    echo "done."
done
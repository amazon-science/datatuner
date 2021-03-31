source ./config.sh

assert_run_dir $PAPER_FOLDER_PATTERN

echo "Running the data formatting for the LDC dataset"
echo $TMP_DATA_FOLDER
python experiments/ldc/preprocess.py --in_folder $TMP_DATA_FOLDER/emnlp2019-dualgraph/process_amr/data/amr_ldc2017t10 --out_folder $DATA_FOLDER/ldc/ --classification_dir $DATA_FOLDER/ldc_consistency

newline

echo "Running the data formatting for the WebNLG dataset"
python experiments/webnlg/preprocess.py --in_folder $TMP_DATA_FOLDER/webnlg/data/v1.4/en/ --out_folder $DATA_FOLDER/webnlg --classification_dir $DATA_FOLDER/webnlg_consistency

newline

echo "Running the data formatting for the ViGGO dataset"
python experiments/viggo/preprocess.py --in_folder $VIGGO_DATA_LOCATION --out_folder $DATA_FOLDER/viggo --classification_dir $DATA_FOLDER/viggo_consistency

newline

echo "Running the data formatting for the Cleaned E2E dataset"
python experiments/e2e/preprocess.py --in_folder $TMP_DATA_FOLDER/e2e-cleaning/cleaned-data/ --out_folder $DATA_FOLDER/e2e --classification_dir $DATA_FOLDER/e2e_consistency

newline
echo "Finished preprocessing the training data"
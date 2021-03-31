source ./config.sh

# Function Definitions

# confirm external dependencies with user
EXTERNAL_DEPS_MSG="""The scripts provided herein will retrieve several third-party libraries,
 environments, and/or other software packages at install-time or build-time (“External Dependencies”)
 from third-party sources.  There are terms and conditions that you need to agree to
 abide by if you choose to install the External Dependencies.  If you do not agree
 with every term and condition associated with the External Dependencies,
 enter “QUIT” in the command line when prompted by the script."""

confirm_external_dependencies() {
  echo
  echo $EXTERNAL_DEPS_MSG
  while true; do
    read -p "Do you want to PROCEED or QUIT? " yn
        case $yn in
        PROCEED)
            echo "Proceeding"
            break
            ;;
        QUIT)
            echo "Quitting"
            exit
            ;;
        esac
    done
}

# clone a specific repository commit to the give folder
clone_repo_commit() {
    # params: repo_url, commit, folder
    git clone $1 $3
    cd $3
    git checkout $2 --quiet
    cd -
}

# check if directory exists and exit with a special message if not
assert_dir_exists() {
    # params: directory, message on failure
    if [ ! -d $1 ]; then
        echo "Error: $1 does not exist."
        echo $2
        exit
    fi
}


#############################################################################################

confirm_external_dependencies

# Check that the LDC2017T10 and ViGGO datasets have been manually downloaded and placed in the correct locations.
assert_dir_exists $LDC2017_DATA_LOCATION "The folder $LDC2017_DATA_LOCATION should contain the LDC2017T10 dataset. Download it from https://catalog.ldc.upenn.edu/LDC2017T10"
newline
assert_dir_exists $VIGGO_DATA_LOCATION "The folder $VIGGO_DATA_LOCATION should contain the ViGGO dataset. Download it from https://nlds.soe.ucsc.edu/viggo"
newline

# Ask the user if the temporary folder exists so that we don't remove and retrieve the data again.
if [ -d $TMP_DATA_FOLDER ]; then
    echo "Directory $TMP_DATA_FOLDER exists. Are you sure you want to delete the data and retrieve it again?"

    select yn in "Yes" "No"; do
        case $yn in
        Yes)
            echo "Alright. Continuing!"
            break
            ;;
        No)
            echo "Exiting"
            exit
            ;;
        esac
    done
else
    echo "Directory $TMP_DATA_FOLDER does not exist."
fi

newline
echo "Creating the folder $TMP_DATA_FOLDER for placing the data there."
rm -rf ./tmp
mkdir -p ./tmp

newline

MAIN_DIR=`pwd`
#############################################################################################

echo "Processing LDC2017T10 dataset"

echo "Getting the repository for data preprocessing"
clone_repo_commit https://github.com/UKPLab/emnlp2019-dualgraph.git 0c58fb7f3ad3b9da3b92b2d2841558807fc79fd0 $TMP_DATA_FOLDER/emnlp2019-dualgraph

echo "Copying the changes needed"
cp ../src/external/ukplab_emnlp2019_dualgraph/split_amr.py $TMP_DATA_FOLDER/emnlp2019-dualgraph/process_amr/split_amr.py
cp ../src/external/ukplab_emnlp2019_dualgraph/gen_LDC2017T10.sh $TMP_DATA_FOLDER/emnlp2019-dualgraph/process_amr/gen_LDC2017T10.sh
cp ../src/external/ukplab_emnlp2019_dualgraph/preproc_amr.py $TMP_DATA_FOLDER/emnlp2019-dualgraph/process_amr/preproc_amr.py
cp ../src/external/ukplab_emnlp2019_dualgraph/preprocess_LDC2017T10.sh $TMP_DATA_FOLDER/emnlp2019-dualgraph/preprocess_LDC2017T10.sh

echo "Running the initial preprocessing"
bash $TMP_DATA_FOLDER/emnlp2019-dualgraph/preprocess_LDC2017T10.sh $LDC2017_DATA_LOCATION ~/

#############################################################################################

newline
cd $MAIN_DIR

# WebNLG dataset
echo "Processing WebNLG dataset"

echo "Retrieving WebNLG data"
clone_repo_commit https://github.com/ThiagoCF05/webnlg.git 12ca34880b225ebd1eb9db07c64e8dd76f7e5784 $TMP_DATA_FOLDER/webnlg

#############################################################################################

newline
cd $MAIN_DIR

# Cleaned E2E
echo "Processing Cleaned E2E dataset"
clone_repo_commit https://github.com/tuetschek/e2e-cleaning.git c6f634ba16aec89f5ec5462e9c62fb3e8c5c5d16 $TMP_DATA_FOLDER/e2e-cleaning

#############################################################################################

cd $MAIN_DIR

# E2E Metrics
echo "Getting E2E metrics repository"
clone_repo_commit https://github.com/tuetschek/e2e-metrics.git dca5d301a97f7264b0827fb5589c0cc51008b5d7 $TMP_DATA_FOLDER/e2e-metrics


newline
echo "Successfully retrieved the data from their sources"

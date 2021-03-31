CONDA_SH_FILE=$1
source $CONDA_SH_FILE


# Confirm external dependencies with user
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

confirm_external_dependencies

echo "Creating the environment"
conda env create --file environment.yml
conda activate finetune

printf "\n"

echo "Downloading the spacy dependenices"
python -m spacy download en_core_web_sm

echo "Downloading the NLTK dependenices"
python -m nltk.downloader punkt

echo "Installing the code in development mode"

printf "\n"

python setup.py develop

printf "\n"

echo "Finished setup"
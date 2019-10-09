#!/bin/bash

. ./env.sh

$R_EXEC -e "packrat::restore()"

# create basic training data
$R_EXEC R/cadr_training_data.R

# create training data for Renton
$R_EXEC preprocess/training_data_RSD.R

# create cohort files
$R_EXEC R/create_cohort_files.R

# clean the cohort files
$R_EXEC R/courses_cohort_cleanup.R

# run the svm model

python -m venv ./python_env

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
	. ./python_env/Scripts/activate
else
	. ./python_env/bin/activate
fi

pip install -r requirements.txt
python ./analyses/svm/svm_cadrs.py

$R_EXEC analyses/cadr_flag_test.R

# TODO:
# run CADR_result_tables.sql?

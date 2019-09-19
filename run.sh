#!/bin/bash

export R_EXEC="Rscript"
export SQLITE_EXEC="sqlite3"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
	: # TODO
elif [[ "$OSTYPE" == "darwin"* ]]; then
	: # TODO
elif [[ "$OSTYPE" == "cygwin" ]]; then
	: # TODO
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
	export PATH=$PATH:"/c/Program Files/R/R-3.4.4/bin/"
	export R_EXEC="Rscript.exe"

	export SQLITE_PATH="/c/Users/jchiu/sqlite-tools-win32-x86-3290000"
	export PATH=$PATH:$SQLITE_PATH
	export SQLITE_EXEC="sqlite3.exe"
else
	: # TODO
fi

# create basic training data
$R_EXEC R/cadr_training_data.R

# create training data for Renton
$R_EXEC preprocess/training_data_RSD.R

# create cohort files
$R_EXEC R/create_cohort_files.R

# clean the cohort files
$R_EXEC R/courses_cohort_cleanup.R

# TODO:
# run the svm model

# TODO:
# run CADR_result_tables.sql

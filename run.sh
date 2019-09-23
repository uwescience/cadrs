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

	export R_EXEC="Rscript.exe"

	which $R_EXEC > /dev/null 2> /dev/null
	if [ $? -ne 0 ]; then
		export PATH=$PATH:"/c/Program Files/R/R-3.4.4/bin/"
	fi

	export SQLITE_EXEC="sqlite3.exe"

	which $SQLITE_EXEC > /dev/null 2> /dev/null
	if [ $? -ne 0 ]; then
		export SQLITE_PATH="/c/Users/jchiu/sqlite-tools-win32-x86-3290000"
		export PATH=$PATH:$SQLITE_PATH
	fi

else
	: # TODO
fi

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
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then

cat > tmp.ps1 <<- EOM
python -m venv .\python_env
.\python_env\Scripts\Activate.ps1
pip install -r requirements.txt
python .\analyses\svm\svm_cadrs.py
EOM
	powershell ./tmp.ps1
	rm tmp.ps1

else
	python -m venv ./python_env
	./python_env/bin/activate
	pip install -r requirements.txt
	python ./analyses/svm/svm_cadrs.py
fi

$R_EXEC analyses/cadr_flag_test.R

# TODO:
# run CADR_result_tables.sql?

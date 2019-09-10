# Analyses Pipeline

## Creating the Training Data 

In order to create the training data that includes Renton School District Course Catalog information run: 

`preprocess/training_data_RSD.R`

You will need the following raw files:

*From OSPI:*
1. `2016-17StateCourseCodes.xlsx`
2. `2015-16-StateCourseCodes.xlsx`
3. `2014_15_StateCourseCodes.csv`

*From Renton:*

1. `rsd_unique_3.csv`

*Technically not raw (Is a variant of `R/cadr_training_data.R`, use file uploaded until I include preprocess code):*

1. `ospi_stud_clean.csv`

## Creating 2017 student cohort to test results
This is ran after the algorithm is trained.

There are two tables created using SQL-script `SQL/CADR_result_tables.sql`

1. `course_2017_cohort.csv` L#8 - L#26
2. `enroll_2017_cohort.csv` L#29 - L#37


## Running Support Vector Machines (SVM) to code CADR courses for 2017 grad cohort

Run: `analyses/svm/avm_cadrs.py`

Description: (more to come)
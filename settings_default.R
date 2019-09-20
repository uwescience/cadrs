
library(here)

#### input paths

gr_hist <- here("data/cadrs/hsCourses.txt")

ospi_crs17_fn <- here("data/cadrs/2016-17StateCourseCodes.xlsx")

ospi_crs16_fn <- here("data/cadrs/2015-16-StateCourseCodes.xlsx")

ospi_crs15_fn <- here("data/cadrs/2014_15_StateCourseCodes.csv")

ospi_crs14_fn <- here("data/cadrs/2013_14_StateCourseCodes.csv")

# renton course catalog
rsd_crs_fn <- "~/data/rsd_unique_3.csv"

dim_school_path <- "~/data/cadr_update/Dim_School.txt"

dim_student_path <- "~/data/cadr_update/Dim_Student.txt"

enrollment_path <- "~/data/cadr_update/enrollments.txt"

#### output paths (including intermediate files)

cadrs_training_path <- here("output/cadrs/cadrs_training.csv")

# cleaned up training set: this is the same file as cadrs_training_path
# above but with some rows hand-filtered out.
# TODO: filtering needs to be put into a script somewhere.
clean_train_fn <- here("data/ospi_stud_clean.csv")

rsd_cadrs_training_path <- "~/data/cadrs/cadrs_training_rsd.csv"

dim_course_path <- here("output/dim_course.csv")

sqlite_database_path <- here("output/ccerCadrDB.db")

course_2017_cohort_path <- "/home/ubuntu/data/db_files/course_2017_cohort.csv" #from SQL

enroll_2017_cohort_path <- "/home/ubuntu/data/db_files/enroll_2017_cohort.csv" #from SQL

course_2017_cohort_clean_path <- "/home/ubuntu/data/db_files/preprocess/course_2017_cohort_clean.csv"

svm_predictions_path <- "/home/joseh/data/svm_cadr_student_predictions_CV.csv"


# TODO: reorganize files in R/ dir into functions and call them from this file
# instead of usine source() to directly execute code

# crs_reshape.R -> calls crs_preprocess.R and post_hs_preprocess.R
# crs_preprocess.R -> creates data/courses_cadrs_text_test.csv
# post_hs_preprocess.R -> creates hs_grads var in R environment used by crs_reshape.R
source("R/crs_reshape.R")

# cadrs_training_data.R -> creates cadrs_dupp_labels.csv
source("R/cadr_training_data.R")

# training_data_setup.R -> creates courses_ospi_b.csv
source("R/training_data_setup.R")

#### python side
# cnn_emb_glove.py -> reads in courses_ospi_b.csv
# crstitle_cosine.py -> reads hsCourses_crsclean.csv
# rnn_model.py -> reads courses_cadrs_text.csv

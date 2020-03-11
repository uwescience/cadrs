#!/usr/bin/env python

# take original training data cadrs_training_rsd.csv
# run some updates and create the subject class labeled set 
# for multi class classification 
# save to google drive for continuous updating
# save copy on repository with updated tags

import pandas as pd 
import csv
import numpy as np
import os
import re
from pathlib import Path

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]

sys.path.append(project_root)
import text_preprocess as tp

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'

# load Json
crs_updates = tp.get_metadata_dict(os.path.join(path_to_metadata, 'mn_crs_updates.json'))
cadr_sub = tp.get_metadata_dict(os.path.join(path_to_metadata, 'cadr_methods.json'))

# load training data rsd
crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'cadrs_training_rsd.csv'), delimiter = ',')

# apply updates from json (cadrs classification fixes (manual)
cadrs = tp.update_data(crs_cat, json_cadr=crs_updates)

# add subject classes for milticlass case
multi = tp.multi_class_df(crs_cat, cadr_sub)

# save file to repository
multi.to_csv(os.path.join(path_to_cadrs, 'training_data_updated.csv'), encoding='utf-8', index=False)
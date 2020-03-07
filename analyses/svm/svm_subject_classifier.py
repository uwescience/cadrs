#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

sys.path.append(project_root)
import text_preprocess as tp

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'
path_to_pretrained_wv = path_root

path_root = '/Users/josehernandez/Documents/eScience/'
path_to_data = path_root + 'data/'
path_to_cadrs = path_root + 'projects/cadrs/'
path_to_metadata = path_to_cadrs + 'metadata/'

crs_cat =  pd.read_csv(os.path.join(path_to_data,'cadrs_training_rsd.csv'), delimiter = ',')

# load Json
crs_updates = tp.get_metadata_dict(os.path.join(path_to_metadata, 'mn_crs_updates.json'))
crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))
cadr_sub = tp.get_metadata_dict(os.path.join(path_to_metadata, 'cadr_methods.json'))

# apply updates from Json
crs_cat['cadr'].describe()
cadrs = tp.update_data(crs_cat, json_cadr=crs_updates)
cadrs.describe()  

# not a lot of coverage from these 
crs_cat[(crs_cat.content_area == 'Business and Marketing') & (crs_cat.cadr == 1)]
crs_cat[(crs_cat.content_area == 'Health Care Sciences') & (crs_cat.cadr == 1)]
crs_cat[(crs_cat.content_area == 'Computer and Information Sciences') & (crs_cat.cadr == 1)]
crs_cat[(crs_cat.content_area == 'Communications and Audio/Visual Technology') & (crs_cat.cadr == 1)]

crs_cat[(crs_cat.content_area == 'Foreign Language and Literature') & (crs_cat.cadr == 1)]


# look at potential classes
crs_cat['content_area'].value_counts()
pd.crosstab(crs_cat.content_area, crs_cat.cadr).sort_values(1, ascending=False)
# Check nulls 
crs_cat["content_area"].isna().mean()
check = crs_cat["content_area"].isna()
crs_cat[check] 
# mostly foreign languge, drop for now and fix if this apporach is promising 
crs_cat.dropna(subset = ["content_area"], inplace=True)
crs_cat = crs_cat.reset_index()

# we want to map the subject categories to something we can manage 
cadr_sub.get("cadr_categories") 
cadr_sub.get("other_cadr") 
cadr_sub.get("non_cadr") 

# map on to new column 

crs_cat["subject_class"] = crs_cat["content_area"]
crs_cat["subject_class"] = crs_cat["content_area"].replace(cadr_sub.get("cadr_categories"))
crs_cat["subject_class"] = crs_cat["subject_class"].replace(cadr_sub.get("other_cadr"))
crs_cat["subject_class"] = crs_cat["subject_class"].replace(cadr_sub.get("non_cadr"))

pd.crosstab(crs_cat.subject_class, crs_cat.cadr).sort_values(1, ascending=False)

# we need to change the non cadr subject specific to non_cadr categories

my_query_index = crs_cat.query('cadr == 0').index
crs_cat.iloc[my_query_index, 8] = "non_cadr"
pd.crosstab(crs_cat.subject_class, crs_cat.cadr).sort_values(1, ascending=False)

# now we can use the subject class as a "classifier"
text =  crs_cat['Name']
labels = crs_cat['subject_class']

num_words = [len(words.split()) for words in text]
max(num_words)

text = text.apply(tp.clean_text)

d = tp.update_data(crs_cat, json_abb=crs_abb) #fix the weird procedure

text = text.replace(to_replace = d, regex=True)

text.apply(lambda x: len(x.split(' '))).sum()
text[1:50]

# beggin algorithm prep
x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state = 42)


###### Pipeline
multinom = Pipeline([
                ('vect', CountVectorizer()), 
                ('tfidf', TfidfTransformer()),
                ('multiclass', MultinomialNB()),
               ])

###### Look at outputs without parameters
multinom.fit(x_train, y_train)
y_pred = multinom.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
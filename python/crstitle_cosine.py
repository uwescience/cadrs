import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sparse_dot_topn import awesome_cossim_topn
import time

pd.set_option('display.max_colwidth', -1)
crs_names =  pd.read_csv('/home/joseh/data/cadrs/hsCourses_crsclean.csv') #renamed
print('The shape: %d x %d' % crs_names.shape)
crs_names.head()

def ngrams(string, n=4):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


course_names = crs_names['crs_copy'].str.strip().unique()
len(course_names)

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

tf_idf_matrix = vectorizer.fit_transform(course_names)

print(tf_idf_matrix[0])
ngrams(course_names[0])

matches = awesome_cossim_topn(tf_idf_matrix, tf_idf_matrix.transpose(), 1000, 0.60)

def get_matches_df(sparse_matrix, name_vector):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similairity': similairity})

matches_df = get_matches_df(matches, course_names)
matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
matches_df.sample(20)
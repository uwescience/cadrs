# Need to use saved model but using right after a fresh run for now
path_root = '/home/joseh/data/'
crs_student =  pd.read_csv(os.path.join(path_root,'crs_2017_cohort.csv'), delimiter = ',', dtype={'Description': str})
crs_student.shape
crs_student.columns

crs_student['Description']=crs_student['Description'].fillna("")

text_out =  crs_student['Description']
num_words_2 = [len(words.split()) for words in text_out]
max(num_words_2)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

text = text_out.apply(clean_text)

text.apply(lambda x: len(x.split(' '))).sum()

text = text.astype(str).values.tolist() # list of text samples

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)


text_tok = pad_sequences(sequences, maxlen=max_seq_len+1)
text_tok.shape


predictions_new = model.predict(text_tok)
len(predictions_new)

pred_cols = pd.DataFrame(predictions_new, columns = ['p_notCADRS', 'p_CADRS'])
pred_cols.head
crs_student.shape

combined_pred = crs_student.merge(pred_cols, left_index=True, right_index=True)
combined_pred.head
combined_pred.to_csv('/home/joseh/data/cnn_cadr_student_predictions.csv', encoding='utf-8', index=False)
print(text_tok[6,124])
max(text_tok)

ans = list(map(max, text_tok))

max(ans)
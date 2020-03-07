import re
import json

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

# clean text 
def clean_text(text):
    # test this out
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    REM_GRADE = re.compile(r'\b[0-9]\w+')
    REPLACE_NUM_RMN = re.compile(r"([0-9]+)|(i[xv]|v?i{0,3})$")    
    text = text.lower()
    text = REM_GRADE.sub('', text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

def update_data(df, json_cadr=None, json_abb=None):
    if json_cadr is not None:
        df['cadr']= df['Name'].map(json_cadr).fillna(df['cadr'])
        return df['cadr']
    if json_abb is not None:
        abb_cleanup = {r'\b{}\b'.format(k):v for k, v in json_abb.items()}
        return abb_cleanup






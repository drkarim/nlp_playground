#%% 
from sklearn.feature_extraction.text import TfidfVectorizer

# The corpus 
corpus = [
'This is the first document.',
'This document is the second document.',
'And this is the third one.',
'Is this the first document?']
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(corpus)



# %%
# TF-IDF to measure of words in documents
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(corpus)

# %%
# checking the td-idf scores of words in the corpus 
import pandas as pd
df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
print(df_tfidf)

# %%

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
''' 
    Implementing Tf-IDF from Scratch 
''' 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

smooth_idf = True
norm_idf = True

# making one hot vector
wc = CountVectorizer()
x = wc.fit_transform(corpus)
wcX = np.array(x.toarray())

# %%
# term frequency
N = wcX.shape[0]
tf = np.array([wcX[i, :] / np.sum(wcX, axis=1)[i] for i in range(N)])

# inverse documents frequency
df = np.count_nonzero(wcX, axis=0)
idf = np.log((1 + N) / (1 + df)) + 1  if smooth_idf else np.log( N / df )


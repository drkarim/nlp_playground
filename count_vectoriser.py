#%% 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np


#%%
cat_in_the_hat_docs=[
       "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
       "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
       "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
       "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
       "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)" 
      ]


# %%
cv = CountVectorizer(cat_in_the_hat_docs)
count_vector=cv.fit_transform(cat_in_the_hat_docs)

# %%
count_vector.shape

# %% 
cv = CountVectorizer(cat_in_the_hat_docs,stop_words=["all","in","the","is","and"])
count_vector=cv.fit_transform(cat_in_the_hat_docs)
count_vector.shape

# %%
cv.stop_words

# %% 
# ignore terms that appeared in less than 2 documents 
cv = CountVectorizer(cat_in_the_hat_docs,min_df=2)
count_vector=cv.fit_transform(cat_in_the_hat_docs)

# %%
cv.vocabulary_

# %%
# ignore terms that appear in 50% of the documents
cv = CountVectorizer(cat_in_the_hat_docs,max_df=0.50)
count_vector=cv.fit_transform(cat_in_the_hat_docs)

# %%
cv.stop_words_

# %%
cv.vocabulary_

# %%
import re
from nltk.stem import PorterStemmer
 
# init stemmer
porter_stemmer=PorterStemmer()
 
def my_cool_preprocessor(text):
    
    text=text.lower() 
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words
    
    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)
 
cv = CountVectorizer(cat_in_the_hat_docs,preprocessor=my_cool_preprocessor)
count_vector=cv.fit_transform(cat_in_the_hat_docs)


# %%
 
# only bigrams, word level
cv = CountVectorizer(cat_in_the_hat_docs,ngram_range=(2,2),preprocessor=my_cool_preprocessor)
count_vector=cv.fit_transform(cat_in_the_hat_docs)

# %%
cv.vocabulary_

# %%

''' 
    Examples of Word Vectoriser in SkLearn 
    Source: https://kavita-ganesan.com/how-to-use-countvectorizer/#.XlEbCJP7RbU



''' 
#%% 
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
cat_in_the_hat_docs=[
       "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
       "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
       "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
       "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
       "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)" 
      ]



# %%
cv = TfidfVectorizer(cat_in_the_hat_docs)
count_vector=cv.fit_transform(cat_in_the_hat_docs)

# %%
cv.vocabulary_

# %%
count_vector.toarray()

# %%
count_vector.shape
# %%
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
 
cv = TfidfVectorizer(cat_in_the_hat_docs,preprocessor=my_cool_preprocessor)
count_vector=cv.fit_transform(cat_in_the_hat_docs)


# %%
cv.vocabulary_

# %%
count_vector.shape

# %%

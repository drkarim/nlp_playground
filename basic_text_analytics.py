''' 
    Tutorial on NLTK 
    Source: https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

'''

#%% 
#Loading NLTK
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords


text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""

#%% 
# Sentence tokeniser 
tokenized_text=sent_tokenize(text)

# word tokeniser 
tokenized_word=word_tokenize(text)

#%% 
# Frequency distribution 
fdist = FreqDist(tokenized_word)

# most common word
fdist.most_common(2)

#%% 
# Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

#%%

# Stop words 
stop_words=set(stopwords.words("english"))
print(stop_words)

#%% 
# filtering stop-words 
filtered_sent=[]
for w in tokenized_word:
    if w.lower() not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

#%% 
# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#%% 
#Lexicon Normalization
#performing stemming and Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

words = ["catching", "flying", "truncate"] 
words_lemmas = [] 
words_stems = []
for w in words: 
    words_lemmas.append(lem.lemmatize(w,pos="v"))
    words_stems.append(stem.stem(w))

print("lemma = ",words_lemmas)
print("stemmer = ",words_stems)


#%%
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
#print(tokens)
nltk.pos_tag(tokens)


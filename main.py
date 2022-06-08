# Import libraries

from unicodedata import name
import pandas as pd
import nltk
import re
import pickle
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Reading data file and combing cols to create new col
df = pd.read_csv("./processed_country.csv", encoding='latin-1')
df["name"] = df["name"].apply(lambda x: x.lower())
df["text"] = df["review"] + " " + df["terrain"] + " " + df["continent"]
df["text"].apply(lambda x: " ".join(re.findall("[a-zA-Z]*", x)).lower())

# tokenize
tokenizer = RegexpTokenizer(r'\w+')
df["tokenized_text"] = df["text"].apply(lambda row: tokenizer.tokenize(row))

# lemmatize 
wnl = nltk.WordNetLemmatizer()
def lem(lst):
    list1=list()
    for i in lst : 
        list1.append(wnl.lemmatize(i))
    return list1

df["lemmatized_text"]=df["tokenized_text"].apply(lambda x : lem(x))


# preparation for stopwords to be used in TfidfVectorizer
my_additional_stop_words = ['acute', 'good', 'great', 'really', 'just', 'nice', 
                            'like', 'day', 'beautiful', 'visit', 'time', 'don',
                            'did', 'place', 'didn', 'did', 'tour', 'sydney','pm', 'the',
                            'lot', '00', 'inside', 'doesn','going','mostly', 'origin',
                            'right', '15']
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, strip_accents='unicode', norm='l2',lowercase=True)

X=[" ".join(text) for text in df["lemmatized_text"].values]
tfidf_matrix=tfidf_vectorizer.fit_transform(X)

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a pandas series with countries as indices and indices as series values 
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# Now model mapping is finished here
with open("matrix_indices.pickle", "wb") as output_file:
    pickle.dump((similarity_matrix, df,indices), output_file)

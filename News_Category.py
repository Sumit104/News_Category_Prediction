
import sys
sys.path.insert(0, 'C:\\Users\\summishra\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\')

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import spacy
import nltk
nlp = spacy.load("en_core_web_sm")

tr_set=pd.read_excel("D://NLP//News//Data_Train.xlsx")
ts_set=pd.read_excel("D://NLP//News//Data_Test.xlsx")

tr_set.head()

tr_set.shape

tr_set.info()

tr_set.SECTION.value_counts()

import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)
    #for w in mytokens:
        #print(w)
        #print(type(w))
        
    mytokens = [ word for word in mytokens if word.pos_ !="VERB" or "ADV" or "DET" or "ADP" or "AUX" or "SYM" or "SPACE"]

    #tags=[VERB, ADP, SYM, NUM]
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    #for w in mytokens:
        #print(w.pos_)
    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    #for w in mytokens:
        #print(w.pos_)
    
    # return preprocessed list of tokens
    return mytokens
#spacy_tokenizer("I am not a apple what do you want he she or your his her go gone went")
# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

#tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
tfidf_vector = TfidfVectorizer(tokenizer=bow_vector)

from sklearn.model_selection import train_test_split

X = tr_set['STORY'] # the features we want to analyze
ylabels = tr_set['SECTION'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('vectorizer_tf', tfidf_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)


from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

predicted1 = pipe.predict(ts_set['STORY'])

pd.DataFrame(predicted1, columns = ['SECTION']).to_excel("D://NLP//News//predictions_N.xlsx")
# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average=None))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average=None))

def MNB():
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()

# Create pipeline using Bag of Words
    pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
    pipe.fit(X_train,y_train)


    from sklearn import metrics
# Predicting with a test dataset
    predicted = pipe.predict(X_test)

    predicted1 = pipe.predict(ts_set['STORY'])

    pd.DataFrame(predicted1, columns = ['SECTION']).to_excel("D://NLP//News//MNB//MNB2.xlsx")
# Model Accuracy
    print("MNB Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("MNB Precision:",metrics.precision_score(y_test, predicted, average=None))
    print("MNB Recall:",metrics.recall_score(y_test, predicted, average=None))

def SGD():
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

# Create pipeline using Bag of Words
    pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
    pipe.fit(X_train,y_train)


    from sklearn import metrics
# Predicting with a test dataset
    predicted = pipe.predict(X_test)

    predicted1 = pipe.predict(ts_set['STORY'])

    pd.DataFrame(predicted1, columns = ['SECTION']).to_excel("D://NLP//News//SGD//SGD1.xlsx")
# Model Accuracy
    print("SGD Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("SGD Precision:",metrics.precision_score(y_test, predicted, average=None))
    print("SGD Recall:",metrics.recall_score(y_test, predicted, average=None))

SGD()
MNB()
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:40:14 2022

@author: Audrey
"""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import string
import re

#Tokenization
def tokenize(filename): 
    tokens = word_tokenize(filename)
    return tokens

#Case Folding
def case_fold(tokens):
    tokens_lower = [token.lower() for token in tokens]
    return tokens_lower
  
#Punctuation Removal
def punct_strip(tokens):
    punct = re.compile('^['+string.punctuation+']$')
    words_nopunct = [word for word in tokens if not re.search(punct,word)]
    return words_nopunct
    
#Stopwords Removal
def nostop(tokens, language):
    stop_words = stopwords.words(language)
    nosw = [word for word in tokens if not word in stop_words]
    return nosw

#Stemming
def stemming(tokens):
    stemmed = [porter.stem(word) for word in tokens]
    return stemmed
    
#Lemmatization

def lemmatization(tokens):
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmas

#All functions

def clean_text(text, normalization):
    tokens = tokenize(text)
    lower = case_fold(tokens)
    nopunct = punct_strip(lower)
    nosw = nostop(nopunct, 'english')
    if normalization == lemmatization:
        final = lemmatization(nosw)
    elif normalization == stemming:
        final = stemming(nosw)
    final = ' '.join(final)
    return final
    

    
sentence = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife"

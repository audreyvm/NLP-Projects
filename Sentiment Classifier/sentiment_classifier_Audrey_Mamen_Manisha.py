# -*- coding: utf-8 -*-
#Audrey, Mamen and Manisha's code
# INSTRUCTIONS: You are responsible for making sure that this script outputs 

# 1) the evaluation scores of your system on the data in CSV_TEST (minimally 
# accuracy, if possible also recall and precision).

# 2) a csv file with the contents of a dataframe built from CSV_TEST that 
# contains 3 columns: the gold labels, your system's predictions, and the texts
# of the reviews.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import string

TRIAL = 0

# ATTENTION! the only change that we are supposed to do to your code
# after submission is to change 'True' to 'False' in the following line:
EVALUATE_ON_DUMMY = False

# the real thing:
CSV_TRAIN = "data/sentiment_train.csv"
CSV_VAL = "data/sentiment_val.csv"
CSV_TEST = "finalresults (1).csv" # you don't have this file; we do

if TRIAL:
    CSV_TRAIN = "data/sentiment_10.csv"
    CSV_VAL = "data/sentiment_10.csv"
    CSV_TEST = "data/sentiment_10.csv"
    print('You are using your SMALL dataset!\n')
elif EVALUATE_ON_DUMMY:
    CSV_TEST = "data/sentiment_dummy_test_set.csv"
    print('You are using the FULL dataset, and using dummy test data! (Ok for system development.)')
else:
    print('You are using the FULL dataset, and testing on the real test data.')
    
    
#below here is our code not theirs#
datatrain = pd.read_csv(CSV_TRAIN)   
dataval= pd.read_csv(CSV_VAL)
datareal=pd.read_csv(CSV_TEST)


######### Pre-processing training data ##########
# lowercase, punctuation and stop words removed #
print('Preprocessing Training Data...\n')
stops = set(stopwords.words('english')+list(string.punctuation))
datatrain['text']=[string.lower() for string in datatrain['text']]
datatrain['cleantext']=datatrain['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
datatrain['tokens']=[string.split(' ') for string in datatrain['cleantext']]
datatrain['vocab']=[set(row) for row in datatrain['tokens']]

########## Inserting Positive and Negative Word Lists and a complete Sentiment Word List#########

with open('positive-words.txt') as file:
    pos_words = file.readlines()
    pos_words = [line.rstrip() for line in pos_words]
    
with open('negative-words.txt') as file:
    neg_words = file.readlines()
    neg_words = [line.rstrip() for line in neg_words]
    
sent_words = pos_words + neg_words
    
########## Features ##########
print('Computing Features...\n')
# Feature one is number of positive words #
datatrain['n_pos_words'] = [np.sum([row.count(positive_word) for positive_word \
                             in pos_words]) for row in datatrain['tokens']]
# Feature two is number of negative words#
datatrain['n_neg_words']=[np.sum([row.count(negative_word) for negative_word \
                              in neg_words]) for row in datatrain['tokens']]
# Feature three is lexical diversity #
datatrain['lex_diversity']=[len(row)for row in datatrain['vocab']]

# Feature four is excluding positive words that come (2 places) after "not" from being classified as positive sentiment #
i=0 
pos = []
for row in datatrain['tokens']:
    pos.append(0)
    for word in pos_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0: 
                if not 'not' == row[position -2]:
                    pos[i]=pos[i] + row.count(word)
    i=i+1
datatrain['n_pos_no_not']=pos

# Feature five is excluding negative words that come (2 places) after "not" from being classified as negative sentiment #
i=0 
neg = []
for row in datatrain['tokens']:
    neg.append(0)
    for word in neg_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0:
                if not 'not' == row[position -2]:
                    neg[i]=neg[i] + row.count(word)
    i=i+1
datatrain['n_neg_no_not']=neg


# feature six is looking only at the last sentiment word in the line to determine whether that is positive or negative
positive_negative = []
positive_negative_list =[]
for row in datatrain['tokens']:
    positive_negative = []
    for word in sent_words:
        if word in row:
            positive_negative=word
    positive_negative_list.append(positive_negative)

datatrain['final_sent_word']= positive_negative_list

final_sent=[]
for row in datatrain['final_sent_word']:
    finalsent=0
    for word in pos_words:
        if word in row:
            finalsent=1
    final_sent.append(finalsent)
datatrain['final_sent']=final_sent     

###############vectorize###############
print('Running training data vectorization...\n')
cv=CountVectorizer()
cv_datatrain=cv.fit_transform(datatrain['cleantext'].values)
lr = LogisticRegression()
y_train=datatrain['sentiment']
lr.fit(cv_datatrain,y_train)
y_pred=lr.predict(cv_datatrain)
y_pred_list=y_pred.tolist()
new_pred_list=[]
for x in y_pred_list:
    if x == 'pos':
        new_pred_list.append(1)
    else:
        new_pred_list.append(0)
datatrain['predicted_by_vectorization']=new_pred_list

print(pd.crosstab(datatrain['sentiment'],datatrain['predicted_by_vectorization']))

      ########## algorithm - logistical regression ##########
print('Running training regression...')
model = LogisticRegression()
X = datatrain[['predicted_by_vectorization']]
y = datatrain['sentiment']
model = model.fit(X, y)
datatrain['predicted_by_logistic_regression'] = model.predict(X)

# =============================================================================
# ######### Training error analysis ##########
# print('Training Data Results...:')
# print(pd.crosstab(datatrain['sentiment'], datatrain['predicted_by_logistic_regression']))
# SK_accuracy=accuracy_score(y,datatrain['predicted_by_logistic_regression'])
# print(f'Accuracy: {SK_accuracy}')
# precision=precision_score(y,datatrain['predicted_by_logistic_regression'], average='binary', pos_label='pos')
# recall=recall_score(y,datatrain['predicted_by_logistic_regression'], average='binary', pos_label='pos')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}\n')
# 
# ########## Reiterated with validation data #########
# 
# ######### Pre-processing training data ##########
# print('Preprocessing Validation Data...\n')
# dataval['text']=[string.lower() for string in dataval['text']]
# dataval['cleantext']=dataval['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
# dataval['tokens']=[nltk.word_tokenize(string) for string in dataval['cleantext']]
# dataval['vocab']=[set(row) for row in dataval['tokens']]
# 
# ########## Features ##########
# print('Computing features...\n')
# dataval['n_pos_words'] = [np.sum([row.count(positive_word) for positive_word \
#                                in pos_words]) for row in dataval['tokens']]
# 
# dataval['n_neg_words']=[np.sum([row.count(negative_word) for negative_word \
#                                   in neg_words]) for row in dataval['tokens']]
# 
# dataval['lex_diversity']=[len(row)for row in dataval['vocab']]
# 
# 
# i=0 
# neg = []
# for row in dataval['tokens']:
#     neg.append(0)
#     for word in neg_words: 
#         if row.count(word) > 0:
#             position= row.index(word)
#             if (position -2) >= 0:
#                 if not 'not' == row[position -2]:
#                     neg[i]=neg[i] + row.count(word)
#     i=i+1
# dataval['n_neg_no_not']=neg
# 
# i=0 
# pos = []
# for row in dataval['tokens']:
#     pos.append(0)
#     for word in pos_words: 
#         if row.count(word) > 0:
#             position= row.index(word)
#             if (position -2) >= 0: 
#                 if not 'not' == row[position -2]:
#                     pos[i]=pos[i] + row.count(word)
#     i=i+1
# dataval['n_pos_no_not']=pos 
# 
# positive_negative = []
# positive_negative_list =[]
# for row in dataval['tokens']:
#     positive_negative = []
#     for word in sent_words:
#         if word in row:
#             positive_negative=word
#     positive_negative_list.append(positive_negative)
# 
# dataval['final_sent_word']= positive_negative_list
# 
# final_sent=[]
# for row in dataval['final_sent_word']:
#     finalsent=0
#     for word in pos_words:
#         if word in row:
#             finalsent=1
#     final_sent.append(finalsent)
# dataval['final_sent']=final_sent    
# 
# ################vectorize################
# print('Running validation data vectorization...\n')
# cv_dataval=cv.transform(dataval['cleantext'].values)
# y_pred=lr.predict(cv_dataval)
# y_pred_list=y_pred.tolist()
# new_pred_list=[]
# for x in y_pred_list:
#     if x == 'pos':
#         new_pred_list.append(1)
#     else:
#         new_pred_list.append(0)
# dataval['predicted_by_vectorization']=new_pred_list
# 
# ########## algorithm - logistical regression ##########
# 
# print('Running validation regression...\n')
# X_val = dataval[['n_pos_words', 'n_neg_words','n_pos_no_not','n_neg_no_not','final_sent','lex_diversity','predicted_by_vectorization']]
# y_val = dataval['sentiment']
# dataval['predicted_by_logistic_regression'] = model.predict(X_val)
# 
# ######### Validation error analysis ##########
# print('Validation Data Results:')
# print(pd.crosstab(dataval['sentiment'], dataval['predicted_by_logistic_regression']))
# SK_accuracy=accuracy_score(y_val,dataval['predicted_by_logistic_regression'])
# print(f'Accuracy: {SK_accuracy}')
# precision=precision_score(y_val,dataval['predicted_by_logistic_regression'], average='binary', pos_label='pos')
# recall=recall_score(y_val,dataval['predicted_by_logistic_regression'], average='binary', pos_label='pos')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}\n')
# 
# =============================================================================
#Final section to run on TEST data#
######### Pre-processing test data ##########
print('Preprocessing Test Data...\n')

datareal['text']=[string.lower() for string in datareal['text']]
datareal['cleantext']=datareal['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
datareal['tokens']=[string.split(' ') for string in datareal['cleantext']]
datareal['vocab']=[set(row) for row in datareal['tokens']]
   
########## Features ##########
print('Computing features...\n')

datareal['n_pos_words'] = [np.sum([row.count(positive_word) for positive_word \
                             in pos_words]) for row in datareal['tokens']]

datareal['n_neg_words']=[np.sum([row.count(negative_word) for negative_word \
                              in neg_words]) for row in datareal['tokens']]

datareal['lex_diversity']=[len(row)for row in datareal['vocab']]

i=0 
pos = []
for row in datareal['tokens']:
    pos.append(0)
    for word in pos_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0: 
                if not 'not' == row[position -2]:
                    pos[i]=pos[i] + row.count(word)
    i=i+1
datareal['n_pos_no_not']=pos

i=0 
neg = []
for row in datareal['tokens']:
    neg.append(0)
    for word in neg_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0:
                if not 'not' == row[position -2]:
                    neg[i]=neg[i] + row.count(word)
    i=i+1
datareal['n_neg_no_not']=neg

positive_negative = []
positive_negative_list =[]
for row in datareal['tokens']:
    positive_negative = []
    for word in sent_words:
        if word in row:
            positive_negative=word
    positive_negative_list.append(positive_negative)

datareal['final_sent_word']= positive_negative_list

final_sent=[]
for row in datareal['final_sent_word']:
    finalsent=0
    for word in pos_words:
        if word in row:
            finalsent=1
    final_sent.append(finalsent)
datareal['final_sent']=final_sent     

################vectorize################
print('Running test data vectorization...\n')
cv_datareal=cv.transform(datareal['cleantext'].values)
y_pred=lr.predict(cv_datareal)
y_pred_list=y_pred.tolist()
new_pred_list=[]
for x in y_pred_list:
    if x == 'pos':
        new_pred_list.append(1)
    else:
        new_pred_list.append(0)
datareal['predicted_by_vectorization']=new_pred_list


########## algorithm - logistical regression ##########
print('Running test data regression...\n')
X = datareal[['predicted_by_vectorization']]
y = datareal['sentiment']
datareal['predicted_by_logistic_regression'] = model.predict(X)

######### Test data Results ##########
print('Finally! The Test Data Results:')
print(pd.crosstab(datareal['sentiment'], datareal['predicted_by_logistic_regression']))
SK_accuracy=accuracy_score(y,datareal['predicted_by_logistic_regression'])
print(f'SK Accuracy logistic regression: {SK_accuracy}')
precision=precision_score(y,datareal['predicted_by_logistic_regression'], average='binary', pos_label='pos')
recall=recall_score(y,datareal['predicted_by_logistic_regression'], average='binary', pos_label='pos')
print(f'Precision Logistic Regression: {precision}')
print(f'Recall Logistic Regression: {recall}')

#Output to CSV#

datareal.to_csv('finalresults.csv',columns=['sentiment','text','predicted_by_logistic_regression'])

gold_neg = y == 'neg'
pred_pos = datareal['predicted_by_logistic_regression'] == 'pos'

gold_pos = y == 'pos'
pred_neg = datareal['predicted_by_logistic_regression'] == 'neg'

datareal['false_positives'] = gold_neg & pred_pos
datareal['false_negatives'] = gold_pos & pred_neg

# We first select our false positives and false negatives by checking where our
# validation gold_label (y_val) has the value 'neg' (gold_neg = y_val == 'neg'). 
# This will return a boolean-vector which contains 'True' everywhere where 
# y_val is equal to 'neg' and 'False' otherwise.
#
# We do the same to find out where our gold-label is positive and where our
# predictions are positive or negative.

# Then we can check in which cases we have false_positives / false_negatives.

twenty_false_positives = datareal.sample(n=20, weights='false_positives')
twenty_false_negatives = datareal.sample(n=20, weights='false_negatives')

# We can now sample 20 random examples of false-positives/false-negatives from 
# our dataframe by using the .sample()-method of pandas-dataframes. Important: 
# we have to indicate which column we want to use as 'weights', otherwise the 
# method will sample from all datapoints.

print('#' * 60)
print('FALSE POSITIVES:')
for example in twenty_false_positives['text']:
    print(example)
print('#' * 60)
print('FALSE NEGATIVES:')
for example in twenty_false_negatives['text']:
    print(example)
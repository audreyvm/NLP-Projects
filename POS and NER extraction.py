import requests
import re
import nltk
from collections import Counter
from bs4 import BeautifulSoup
import es_core_news_sm 
nlp = es_core_news_sm.load()
from cleaning_functions import *

#get & preprocess article
blacklist = ['script', 'style', 'aside']
def url_to_string(url):
    request = requests.get(url)
    html = request.text
    soup = BeautifulSoup(html, "html5lib")

    for script in soup(blacklist):
        script.extract()
    text = [paragraph.get_text() for paragraph in soup.find_all('p')]
    return " ".join(text)

url = "https://es.wikipedia.org/wiki/D%C3%ADa_de_la_Marmota"
article = url_to_string(url)
article = re.sub('(\n|\s+)',' ', article)
article = re.sub('(\\u200b|\[[0-9]\])', '', article)

tokenized = tokenize(article)
doc = nlp(' '.join(tokenized))

#extracct most common entity types
labels = [x.label_ for x in doc.ents]
labels_dict = Counter(labels)
for label in labels_dict.keys():
    print(label)
    entities = [x.text for x in doc.ents if x.label_ == label]
    print(Counter(entities).most_common(5))

#extract most common nouns and verbs
docnosw = [word for word in doc if not word.is_stop and not word.is_punct]

def get_most_common_category(doc,category,top_n):
  words_category = [word.lemma_ for word in doc if word.pos_ == category]
  words_category_dict = Counter(words_category)
  most_common = [word for word, count in words_category_dict.most_common(top_n)]
  return most_common

print('Most common nouns: ', get_most_common_category(docnosw, 'NOUN', 5))
print('Most common verbs: ',get_most_common_category(docnosw, 'VERB', 5))



# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 23:42:32 2015

@author: Gaston Besanson
"""
#Library Used
from gensim import corpora, models, similarities
import codecs
import scipy
import os, json
import pandas as pd
import numpy as np
import nltk
import re
from sklearn import feature_extraction
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import gensim
import sys

# Setting utf8 as default coding
reload(sys)  
sys.setdefaultencoding('utf8')

# List to where the Json Files would be imported
contents = []

# Path where the JSON FIles are 
path_to_json = 'C:/Users/gaston/Documents/SHARE_VM/Comments/'

# Searching for JSON Files
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print json_files  # this prints the name of the json file

# Appending the JSON files to the list
for js in json_files:
    with open(os.path.join(path_to_json, js)) as json_file:
        contents.append(json.load(json_file))
        print js

# Turning into a Data Frame
df=pd.DataFrame.from_dict(contents)

# Some facts from the data

df['city'].value_counts()

df['lawfirm'].value_counts()

df['pages'].value_counts()

# Just a Data Frame with the text in each comment
justText=df["text"]

# Drop text that is empty
justText=justText.dropna()

# Drop text that is duplicated
justText=justText.drop_duplicates()

# All words in lower cases
justText=justText.str.lower()

# Replacing "\u2019" by '
justText=justText.replace({u"\u2019": "'"}, regex=True)

# Remove non-breaking spaces
justText=justText.replace({u"\u00A0": " "}, regex=True)

# Turn the data frame into vector
textComments=justText.values

# Stopwords
stopwords = nltk.corpus.stopwords.words('english')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

#  Remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# list for tokenized documents in loop
texts = []

# loop through document list
for i in textComments:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stopwords]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# Documents into a id - term dictionary
dictionary = corpora.Dictionary(texts)

#remove extremes tokens
dictionary.filter_extremes(no_below=1, no_above=0.8)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=10)

# Save LDA model
ldamodel.save('completeCorpus.lda')

# Print the top 15 words in the 20 topics
for i in range(0, 20):
    temp = ldamodel.show_topic(i,  15)
    terms = []
    for term in temp:
        terms.append(term)
    print "Top 15 terms for topic #" + str(i+1) + ": "+ ", ".join([i[1] for i in terms])

###########################################################

#Hand picking topics Related to the technical, legal, economic and ideological discussions

#The following Topics make sense in the context
#Topics # 4, 5, 7, 9 , 14, 17, 19

# In the print Topic 4 appears as Topic 3, because of the zero index in Python

###########################################################

#Query 1

query = "Internet was born free, it should remind this way."

# Treatment of the query 
query = query.lower()
query = tokenizer.tokenize(query)
query = [i for i in query if not i in stopwords]
query = [p_stemmer.stem(i) for i in query]

# To be able to search on the index for the new query
id2word = gensim.corpora.Dictionary()
_ = id2word.merge_with(dictionary.id2token)

# Trasnform the query into an index
query = id2word.doc2bow(query)

# Search the query in the model
ldamodel[query]

# Return the topics orderer from less likely to the more related topic
a = list(sorted(ldamodel[query], key=lambda x: x[1]))
print(a[0])
print(a[-1])

# print the words of the more related topic
ldamodel.print_topic(a[-1][0])

#Query 2

query = "a non-neutral environment could kill the innovation in the Internet ecosystem."

# Treatment of the query 
query = query.lower()
query = tokenizer.tokenize(query)
query = [i for i in query if not i in stopwords]
query = [p_stemmer.stem(i) for i in query]

# To be able to search on the index for the new query
id2word = gensim.corpora.Dictionary()
_ = id2word.merge_with(dictionary.id2token)

# Trasnform the query into an index
query = id2word.doc2bow(query)

# Search the query in the model
ldamodel[query]

# Return the topics orderer from less likely to the more related topic
a = list(sorted(ldamodel[query], key=lambda x: x[1]))
print(a[0])
print(a[-1])

# print the words of the more related topic
ldamodel.print_topic(a[-1][0])

#Query 3

query = "Some services are built on top of our networks and their traffic obligate us to invest in infrastructure plus they compete with our own products."

# Treatment of the query 
query = query.lower()
query = tokenizer.tokenize(query)
query = [i for i in query if not i in stopwords]
query = [p_stemmer.stem(i) for i in query]

# To be able to search on the index for the new query
id2word = gensim.corpora.Dictionary()
_ = id2word.merge_with(dictionary.id2token)

# Trasnform the query into an index
query = id2word.doc2bow(query)

# Search the query in the model
ldamodel[query]

# Return the topics orderer from less likely to the more related topic
a = list(sorted(ldamodel[query], key=lambda x: x[1]))
print(a[0])
print(a[-1])

# print the words of the more related topic
ldamodel.print_topic(a[-1][0])

#Query 4

query = "Across the Internet, data is transmitted as packets and depending on the data size more \
 or less amount of packets are used to represent that data. As mention before, Net Neutrality establish \
 that this packets have to be treated in an equal way in terms of price and service. \
 The raise of the discussion on Net Neutrality is related to the \
emerge of two technology driven phenomenon: (i) Now we are able \
to distinguish among packets; (ii) the new wave of applications and \
contents, which are sensible to delays, that incremented the demand \
of traffic management over the network. \
This mass of new traffic given by these new functionalities and online services is \
totally asymmetric between ISPs, mainly due to some \
prominent and resource consuming content providers which are usually \
 connected to a single ISP. As Boussion et al. (2012) explains with \
a Youtube example. This site is accessed by all users while hosted \
by a single Tier 1 ISP, and whose traffic now constitutes a non-negligible \
part of the whole Internet traffic. \
This asymmetry in the flow of traffic, reveal a flaw in the ISP's \
business model, where it charge both end users and content providers \
directly connected to them, and have public peering or transit agreements with other ISPs. \
Leaving out the possibility of earning income from traffic related to content providers that are \
associated with other ISPs. And not being able to have a share of this content providers' \
online advertising revenue may perverse the incentives to invest in \
network infrastructure that some ISPs have, if this means that only \
the content providers benefit from it. \
The ISP market, is not a traditional market and it is defined as a two-sided market \
On one side the ISP deals with *content providers* and in the other there are the users, \
which consume these contents; in other words, the ISP facilitates the interaction among both sides \
Net Neutrality implied two rules: (i) *zero-price* and (ii) *non-discrimination*. \
The first one means that no charge can be imposed to the originator of data packets for transmitting \
them to users -what is call a *termination fee*-. \
And the second one, is that ISP cannot engage in traffic management by favoring certain packets over others."


# Treatment of the query 
query = query.lower()
query = tokenizer.tokenize(query)
query = [i for i in query if not i in stopwords]
query = [p_stemmer.stem(i) for i in query]


# To be able to search on the index for the new query
id2word = gensim.corpora.Dictionary()
_ = id2word.merge_with(dictionary.id2token)

# Trasnform the query into an index
query = id2word.doc2bow(query)

# Search the query in the model
ldamodel[query]

# Return the topics orderer from less likely to the more related topic
a = list(sorted(ldamodel[query], key=lambda x: x[1]))
print(a[0])
print(a[-1])

# print the words of the more related topic
ldamodel.print_topic(a[-1][0])
 


ldamodel.show_topic(1, 100)


# coding: utf-8

# # NLP Tweets - Trump vs Clint

# This is my initiation with NLP so let's see how it goes..
# Here I will be working with a set of about 3000 recent tweets by two US presidential candidates, Hillary Clinton and Donald Trump. The dataset contains additional metadata, feel free to do some extra digging.

# ## 1. Reading Data
# 
# >The first step in any machine learning task is concerned with reading in the data. 
# In order to understand how to read in the data, I took a look at how it is stored and the tweets are stored in a comma-separated (csv) file. 

# In[1]:

import pandas as pd


# In[2]:

# Reading the data as follows:
header = ['id', 'handle', 'text', 'is_retweet', 'original_author', 'time', 'in_reply_to_screen_name', 
          'in_reply_to_status_id', 'in_reply_to_user_id', 'is_quote_status', 'lang', 'retweet_count', 
          'favorite_count', 'longitude', 'latitude', 'place_id', 'place_full_name', 'place_name', 'place_type', 
          'place_country_code', 'place_country', 'place_contained_within', 'place_attributes', 'place_bounding_box', 
          'source_url', 'truncated', 'entities', 'extended_entities']
df = pd.read_csv('tweets.csv', sep=',', skiprows=1, names=header)


# >Most of the columns in this file contain meta-information such as time stamps, number of retweets, favourites, etc. The `skiprows` argument is used to skip the first row in the file because it is header information.
# 
# >The most relevant columns at this point are **id** (unique id's of the tweets), **handle** (Clint and Hillary tweets'), and **text** containing the tweets' text.

# In[3]:

# Using unique() to remove duplicates
ids = df.id.unique() 
n_ids = ids.shape[0]
politicians = df.handle.unique()
n_politicians = politicians.shape[0]
tweets = df.text.unique()
n_tweets = tweets.shape[0]
print ('Number of politicians = ' + str(n_politicians) 
       + '\n' + 'Who: ' + str(politicians)
       + '\n' + 'Overall number of tweets = ' + str(n_ids) 
       + '\n' + 'Number of unique tweets = ' + str(n_tweets)) 


# ## 2. Data Preprocessing
# 
# >These will be my initial steps:
# >* Converting strings to lowercase
# >* Split strings of text into separate words
# >* Lemmatise words

# ***Importing NLTK, word tokenizer and word lemmatizer:***

# In[4]:

import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# ***Preprocessing textual data with preprocess method:***

# In[5]:

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    return [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(unicode(sentence, errors='ignore')) 
            if not word.startswith('/')]


# > Note that filtering out the words starting with "/" is a straightforward solution for excluding the urls from the set of words (these are frequently included in tweets, but most probably will not provide particular useful information for any future analysis). 

# ***Defining a method to extract the text of each tweet and store it to each candidate:***

# In[6]:

def extract(politician_name):
    tweet_texts = []
    for i in range(0, n_ids):
        author = df.handle[i]
        tweet = df.text[i]
        if author==politician_name:
            tweet_texts.append(tweet.strip())
    return tweet_texts


# ***Extracting tweets from Trump and Clint and storing them as different lists:***

# In[7]:

# Exploring text:
trump = extract('realDonaldTrump')
clint = extract('HillaryClinton')
print 'Number of Clint tweets:', len(clint)
print 'Number of Trump tweets:', len(trump)
# Example of trump text: 
for text in trump:
    print '\nTrump Original Text: ', text
    print 'Trump Preprocessed Text: ', preprocess(text)


# > The dataset is quite balanced between the two candidates. Trump contains 3218 tweets and Clint has 3226 tweets.

# ## 3. Data Inspection/Understanding

# ***Saving the different tweets in different lists:***

# In[8]:

trump = extract('realDonaldTrump')
clinton = extract('HillaryClinton')


# ***Importing libraris Text and FreqDist from nltk: ***

# In[9]:

from nltk.text import Text
from nltk import FreqDist


# ***Creating the `context_search` method to search for specific words and their contexts in the tweets:***

# In[10]:

def context_search(a_list, search_word):
    for tweet in a_list:
        word_list = preprocess(tweet)
        text_list = Text(word_list)
        search_word = search_word.lower()
        if search_word in word_list:
            text_list.concordance(search_word)


# ***Exploring the use of the word 'America' by both Trump and Clinton:***

# In[11]:

context_search(clinton, 'America')


# In[12]:

context_search(trump, 'America')


# ***Checking for frequency of selected words 'america' and 'russia' and 'gun':***

# In[13]:

def search_word_freq (a_list, word):
    word_list = []
    for tweet in a_list:
        word_list += preprocess(tweet)
    text_list = Text(word_list)
    fdist = FreqDist(text_list)
    print ("Frequent of '%s' Word Search: ") %word + str(fdist[word])

print "Trump:"
search_word_freq(trump,'america')
search_word_freq(trump,'russia')
search_word_freq(trump,'gun')
print "Clint:"
search_word_freq(clint, 'america')
search_word_freq(clint, 'russia')
search_word_freq(clint, 'gun')


# ***Checking the vocabulary for each candidate:***

# In[14]:

def vocabulary(a_list):
    vocab = []
    for tweet in a_list:
        text_list = preprocess(tweet)
        vocab = vocab + text_list
    return sorted(set(vocab))

trump_vocab = vocabulary(trump)
clinton_vocab = vocabulary(clinton)
print ("Trump's vocabulary size: ") + str(len(trump_vocab))
print ("Clinton's vocabulary size: " + str(len(clinton_vocab)))


# ***Checking how diverse and lexically rich is the language of both candidates:***

# In[15]:

def word_count(a_list):
    total_length = 0
    unique_words = []
    for tweet in a_list:
        text_list = preprocess(tweet)
        total_length = total_length + len(text_list)
        unique_words = unique_words + text_list
    print ("Total length " + str(total_length))
    print ("Unique words " + str(len(set(unique_words))))
    print ("Lexical richness " + str(100 * round(float(len(set(unique_words)))/float(total_length), 4)) + "%")

print '\nTrump:'
word_count(trump)
print '\nClinton:'
word_count(clinton)


# ***Extracting stopwords, punctuation, filtering words and ploting frequent words:***

# In[16]:

from nltk.corpus import stopwords
import re
import string

stoplist = stopwords.words('english')

def filtered_words(a_list):
    word_list = []
    for tweet in a_list:
        word_list += preprocess(tweet) # preprocessing tweets
    text_list = Text(word_list)
    text_with_no_stopwords = [] # removing stopwords
    for word in text_list:
        if word not in stoplist:
            text_with_no_stopwords.append(word)
    return [w for w in text_with_no_stopwords if len(w)>2 and w not in string.punctuation] # removing punctuation 
    
def extract_freq_words (a_list):
    get_freq = filtered_words(a_list)
    fdist = FreqDist(get_freq) # getting frequency
    print(fdist.most_common(30)) 
    fdist.plot(30, cumulative=True)

print "Trump:"
extract_freq_words(trump)
print "Clint:"
extract_freq_words(clinton)


# Interesting information when compared with other previous presidents reports from `https://www.cmu.edu/news/stories/archives/2016/march/speechifying.html`.
# <img style="max-width:100%; width: 70%; max-width: none" src="img1.jpg" >

# ## 4. Supervised Learning: Predicting Tweet Author
# 
# Predicting the author of a tweet by using a supervised method: Naive Bayes `(http://scikit-learn.org/stable/)`. 

# In[17]:

import random
from nltk import NaiveBayesClassifier, classify


# ***Creating features from textual data, words used in tweets:***

# In[18]:

stoplist = stopwords.words('english')
    
def get_features(tweet, setting): 
    features = {}
    word_list = preprocess(tweet)
    for word in word_list:
        if word not in stoplist:
            if setting=="bow":
                features[word] = features.get(word, 0) + 1
            else:
                features[word] = True
    return features


# ***Defining a train method and training the classifier Naive Bayes:*** 

# In[19]:

def train(features, n): # n is the size of the training set, the proportion of tweets to use for training, e.g. 80%
    train_size = int(len(features) * n)
    
    # Creating training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ("Training set size = " + str(len(train_set)) + " tweets")
    print ("Test set size = " + str(len(test_set)) + " tweets")
    
    # Training the Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier


# ***Extracting features and training the classifier:***

# In[20]:

# Getting the tweets
all_tweets = [(tweet, "trump") for tweet in trump] 
all_tweets += [(tweet, "clinton") for tweet in clinton]
random.shuffle(all_tweets) # shuffling the data for a better randomness
print ("Dataset Size = " + str(len(all_tweets)) + " tweets")
    
# Extract the textual features
all_features = [(get_features(tweet, ""), label) for (tweet, label) in all_tweets]
print ("Collected = " + str(len(all_features)) + " features")

# Train the classifier with a 80:20 split
train_set, test_set, classifier = train(all_features, 0.8)


# ***Evaluating the accuracy of the classifier and extracting the most predictive features/most informative words:***

# In[21]:

def evaluate(train_set, test_set, classifier):
    print ("Traning Set Accuracy = " + str(classify.accuracy(classifier, train_set)))
    print ("Test Set Accuracy = " + str(classify.accuracy(classifier, test_set)))    
    # checking which words are most informative for the classifier – randomly selected 50 most informative
    classifier.show_most_informative_features(50)
    
# evaluating classifier's performance  
evaluate(train_set, test_set, classifier)


# Depending on the random split and the random shuffle before splitting we can expect around 98% accuracy on the training and 92-94% on the test data. 
# 
# The most informative words characterising Trump include, among others, many adjectives and emotionally strong words like *wow* and *fantastic*, or mentions *rubio* and *cruz*. On the other side, characterising Clinton include, for example, *gold*, *fear*, *demsinphilly*, *progress* and *violence*. 

# ***Evaluating the performance of the classifier on each of the two classes separately:***
# 
# Spliting the test set into *trump* and *clinton* subsets and calculating each subset accuracy separately.

# In[22]:

def train_class (features, n, label): # n is the proportion of tweets to use for training; label is the candidate
    train_size = int(len(features) * n)
    
    # Initialise Training and Test sets
    train_set = features[:train_size] 
    test_set = []
    for feature in features[train_size:]:
        if feature[1]==label:
            test_set.append(feature)
    print ("Training Size = " + str(len(train_set)) + " tweets")
    print ("Test Size = " + str(len(test_set)) + " tweets")
    
    # Train the Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier


# ***Testing and evaluating the classifier for each class:***

# In[23]:

# Getting the tweets 
all_tweets = [(tweet, "trump") for tweet in trump] 
all_tweets += [(tweet, "clinton") for tweet in clinton]
random.shuffle(all_tweets)
print ("Dataset size = " + str(len(all_tweets)) + " tweets")
    
# Extracting the textual features
all_features = [(get_features(tweet, ""), label) for (tweet, label) in all_tweets]
print ("Collected " + str(len(all_features)) + " features")

# Training Classifier with 80% and Test on "trump"
print ("\nAccuracy on tweets from Trump:")
train_set, test_set, classifier = train_class(all_features, 0.8, "trump")
evaluate(train_set, test_set, classifier)
# Training Classifier with 80% and Test on "clinton"
print ("\nAccuracy on tweets from Clinton:")
train_set, test_set, classifier = train_class(all_features, 0.8, "clinton")
evaluate(train_set, test_set, classifier)


# **Trump:**
# + Training Accuray: ~ 98.2%
# + Test Accuracy: ~ 90.3%
# 
# **Clint:**
# + Training Accuracy: ~ 98.2%
# + Test Accuracy: ~ 95.7%
# 
# Depending on the data random split selected these results will change.

# ## 5. Unsupervised Learning: Predicting Tweet Author
# 
# Now will use a Unsupervised Method KMeans clustering `(http://scikit-learn.org/stable/)`.

# In[24]:

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics


# ***Initializing the data by storing the tweets in `data` data-structure and candidates names in the list of `labels`. Casting the strings *trump* and *clinton* to numerical values, for example, 0 and 1:***

# In[25]:

def init_clust_data():
    all_tweets = [(tweet, "trump") for tweet in trump] 
    all_tweets += [(tweet, "clinton") for tweet in clinton]
    random.shuffle(all_tweets) 
    data = []
    labels = []
    for entry in all_tweets:
        data.append(entry[0])
        if entry[1]=="trump":
            labels.append(0)
        else:
            labels.append(1)
        
    print("Data:")
    print(str(len(data)) + " tweets in 2 categories\n")
    print(data)

    print("Data labels: ")
    print(labels)
    gs_clusters = len(set(labels))
    print("Gold standard clusters: " + str(gs_clusters))
    return data, labels, gs_clusters


data, labels, gs_clusters = init_clust_data()


# ***Applying data transformation and dimensionality reduction:***

# In[26]:

def transform(orig_data, orig_dim, red_dim):
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, 
                                 max_features=orig_dim,
                                 stop_words='english',
                                 use_idf=True)
    trans_data = vectorizer.fit_transform(orig_data)

    print("\nData contains: " + str(trans_data.shape[0]) + " with " + str(trans_data.shape[1]) + " features.")

    svd = TruncatedSVD(red_dim)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    
    return lsa.fit_transform(trans_data), vectorizer, svd

trans_data, vectorizer, svd = transform(data, 10000, 300)


# ***Applying KMeans Clustering Algorithm:***

# In[27]:

def cluster(trans_data, gs_clusters):
    km = KMeans(n_clusters=gs_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(trans_data)
    return km

km = cluster(trans_data, gs_clusters)


# ***Evaluating the identified clusters against gold standard clusters:***

# In[28]:

def evaluate(km, labels, svd):
    print("Clustering report:\n")
    print("* Homogeneity: " + str(metrics.homogeneity_score(labels, km.labels_)))
    print("* Completeness: " + str(metrics.completeness_score(labels, km.labels_)))
    print("* V-measure: " + str(metrics.v_measure_score(labels, km.labels_)))

    print("\nMost discriminative words per cluster:")
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(gs_clusters):
        print("Cluster " + str(i) + ": ")
        cl_terms = ""
        for ind in order_centroids[i, :50]:
            cl_terms += terms[ind] + " "
        print(cl_terms + "\n")
        

evaluate(km, labels, svd)


# The results suggest that unsupervised approach performs worse than supervised algorithm.

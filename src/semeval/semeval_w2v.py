import torch
import h5py
import scipy as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import pandas as pd
import sys
import json
import pickle
from nltk import word_tokenize
from pprint import pprint as pp

import operator
import csv
# from textcleaner import tokenize_by_word
from credbankprocessor import preprocessing_tweet_text
from preprocessing import text_preprocessor

from ast import literal_eval # convert string list to actual list
from nltk import TweetTokenizer

from semeval_data_processor import load_csv
from simsem_eval import eval
import gensim
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import string
_stop_words = stopwords.words('english')
stop_words_filter = lambda t : filter(lambda a: a not in _stop_words, t)
punctuation_filter = lambda t : filter(lambda a: a not in string.punctuation, t)

lemmatizer = WordNetLemmatizer()
postag_map = {'j': 'a','n': 'n','v': 'v'}

def _lemmatize(token_list):
    lemmatized_tokens = [lemmatizer.lemmatize(token, postag_map[tag[0].lower()]) if tag[0].lower() in ['j', 'n', 'v']
                        else lemmatizer.lemmatize(token) for token, tag in nltk.pos_tag(token_list)]
    return lemmatized_tokens



pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None



# batch_num = sys.argv[1]
batch_num = 1
batch_size = 1000

def load_data():
    # data_type = sys.argv[1]
    data_type = "merged_semeval"
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data/{}.csv'.format(data_type))
    print("data path ", data_path)
    data = load_csv(data_path)
    print(len(data))
    data.drop(['Unnamed: 0'], inplace=True, axis=1)
    # data.dropna(inplace=True)
    # data = data[:1001]
    print(data.isnull().any().any())
    if data.isnull().any().any():
        raise ValueError
    print("the number of rows in the original data: ", len(data))

    return data

def load_w2v():
    w2v_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data/word2vec/ord2')
    model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    return model

def get_batch_embeddings(empty_batch_embeddings, tokenised_doc, embeddings):
    """
    Compute embeddings of a batch
    :param empty_batch_embeddings: empty numpy array which defines the shape of output
    :param tokenised_doc: a set of tokenised tweets
    :param embeddings: pre-trained glove embeddings
    :return: numpy array (number of tweets in the batch, glove vector dimension)
    """
    batch_embeddings = empty_batch_embeddings
    for tweet in tokenised_doc:
        tweet_embeddings = np.empty((0, 300))
        for token in tweet:
            if token in embeddings:
                tweet_embeddings = np.mean(np.vstack((tweet_embeddings, embeddings[token].reshape(1,-1))), axis=0)
                tweet_embeddings = tweet_embeddings.reshape(1, -1)

        if tweet_embeddings.shape[0] == 0: # assign an array filled with zeros to an empty embedding
            tweet_embeddings = np.zeros((1,300))
        batch_embeddings = np.vstack((batch_embeddings, tweet_embeddings))
    return batch_embeddings

def w2v_semantic_similarity():
    w2v_embeddings = load_w2v()
    data = load_data()
    print(w2v_embeddings['user'].shape)

    sim_scores = []
    for j in range(0, len(data), batch_size):
        batch = data.loc[j:j + batch_size - 1]
        start = time.time()
        print("")

        tokenised_tweet1 = list(map(lambda x: literal_eval(x), batch['processed_tweet1'].values))
        tokenised_tweet1 = list(map(lambda x: list(stop_words_filter(x)), tokenised_tweet1))
        tokenised_tweet1 = list(map(lambda x: list(punctuation_filter(x)), tokenised_tweet1))
        tokenised_tweet1 = list(map(lambda x: _lemmatize(x), tokenised_tweet1))

        tokenised_tweet2 = list(map(lambda x: literal_eval(x), batch['processed_tweet2'].values))
        tokenised_tweet2 = list(map(lambda x: list(stop_words_filter(x)), tokenised_tweet2))
        tokenised_tweet2 = list(map(lambda x: list(punctuation_filter(x)), tokenised_tweet2))
        tokenised_tweet2 = list(map(lambda x: _lemmatize(x), tokenised_tweet2))

        batch_embeddings = np.empty((0, 300))
        batch_embeddings_tweet1 = get_batch_embeddings(batch_embeddings, tokenised_tweet1, w2v_embeddings)
        batch_embeddings_tweet2 = get_batch_embeddings(batch_embeddings, tokenised_tweet2, w2v_embeddings)

        print(batch_embeddings_tweet1.shape)
        print(batch_embeddings_tweet2.shape)

        assert batch_embeddings_tweet1.shape[0] == batch_embeddings_tweet2.shape[0] == len(batch)

        print("Averaged w2v vector dimension (tweet1): ", batch_embeddings_tweet1.shape)
        print("Averaged w2v vector dimension (tweet2): ", batch_embeddings_tweet2.shape)

        batch_scores = []
        for i in range(batch_embeddings_tweet1.shape[0]):
            score = cosine_similarity(batch_embeddings_tweet1[i, :].reshape(1, -1), batch_embeddings_tweet2[i, :].reshape(1, -1))
            sim_scores.append(score.flatten()[0])
            batch_scores.append(score.flatten()[0])
        end = time.time()
        print("Time elapsed for bath {}: {}".format(int(j / batch_size), end - start))
        batch['sim_scores'] = batch_scores
        # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
        #                        'data/semeval2015/results/{}_batch{}.csv'.format('w2v_merged', int(j / batch_size)))
        # batch.to_csv(outfile)
    data['sim_scores'] = sim_scores
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data/semeval2015/results/word2vec+preprocessing/{}.csv'.format('w2v_merged'))
    data.to_csv(outfile)
    print("Done")

def eval_results():
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/results/word2vec+preprocessing{}.csv'.format('w2v_merged'))
    print(result_path)
    eval(result_path)

# w2v_semantic_similarity()
eval_results()

# outfile = os.path.join(os.path.dirname(__file__), '..', '..','..','Data','glove.twitter.27B')
# outfile_abs_path =os.path.abspath(outfile)
# # with open(os.path.join(outfile_abs_path, 'glove_200d_dict.pickle'), 'wb') as f:
# #     pickle.dump(embeddings_index, f)
#
# with open(os.path.join(outfile_abs_path, 'glove_200d_dict.pickle'), 'rb') as f:
#     x = pickle.load(f)
#     print(type(x))
# # print(x)
# #
# for key, value in x.items():
#     print(key, value)
#     raise SystemExit




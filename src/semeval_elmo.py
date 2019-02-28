import torch
import allennlp
import h5py
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.commands.elmo import ElmoEmbedder

# import tensorflow_hub as hub
# import tensorflow as tf
from allennlp.data.dataset_readers import DatasetReader

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

def elmo_semantic_similarity():
    """
    Compute semantic similarity using ELMo embeddings
    :return:
    """
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    elmo = ElmoEmbedder(options_file= options_file, weight_file=weight_file)
    data = load_data()

    sim_scores =[]
    for j in range(0, len(data), batch_size):
        batch = data.loc[j:j+batch_size-1]
        start = time.time()

        tokenised_tweet1 = list(map(lambda x: literal_eval(x), batch['processed_tweet1'].values))
        tokenised_tweet2 = list(map(lambda x: literal_eval(x), batch['processed_tweet2'].values))

        #######################
        # TO DO: Apply bactch #
        #######################
        ## Computes the ELMo embeddings for a batch of tokenized sentences.
        elmo_tweet1 = elmo.embed_batch(tokenised_tweet1) # a list of tensors
        print("elmo tweet1 shape ", elmo_tweet1[2].shape)
        elmo_tweet2 = elmo.embed_batch(tokenised_tweet2)

        ## Compute the mean elmo vector for each tweet
        elmo_tweet1_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_tweet1))
        print("elmo tweet1 avg shape ", elmo_tweet1_avg[0].shape)
        elmo_tweet2_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_tweet2))
        elmo_tweet1_avg = np.squeeze(np.asarray(elmo_tweet1_avg), axis=1)
        elmo_tweet2_avg = np.squeeze(np.asarray(elmo_tweet2_avg), axis=1)
        print("Averaged ELMo vector dimension (tweet1): ", elmo_tweet1_avg.shape)
        print("Averaged ELMo vector dimension (tweet2): ", elmo_tweet2_avg.shape)
        assert elmo_tweet1_avg.shape[0] == elmo_tweet2_avg.shape[0] == len(batch)

        batch_scores =[]
        for i in range(elmo_tweet1_avg.shape[0]):
            score = cosine_similarity(elmo_tweet1_avg[i,:].reshape(1,-1), elmo_tweet2_avg[i,:].reshape(1,-1))
            # sim_scores[i] = score.flatten()[0]
            sim_scores.append(score.flatten()[0])
            batch_scores.append(score.flatten()[0])
        end = time.time()
        print("Time elapsed for bath {}: {}".format(int(j/batch_size), end-start))
        batch['sim_scores'] = batch_scores
        # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
        #                        'data/semeval2015/{}_batch{}.csv'.format('elmo_merged', int(j/batch_size)))
        # batch.to_csv(outfile)
    data['sim_scores'] = sim_scores
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/results/elmo/{}.csv'.format('elmo_merged_55b'))
    data.to_csv(outfile)
    print("Done")

def eval_results():
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/results/glove/{}.csv'.format('elmo_merged_55b'))
    eval(result_path)

elmo_semantic_similarity()
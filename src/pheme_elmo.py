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
from semeval_data_processor import load_csv
from simsem_eval import eval

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None



# batch_num = sys.argv[1]
batch_num = 1
batch_size = 1000

def load_data():
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data_augmentation/data/ref/boston_{}.csv'.format(batch_num))

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data_augmentation/data/candidates/boston_candidates_rts_more-than-4-tokens.csv')

    ref = load_csv(ref_path)
    data = load_csv(data_path)
    print(len(data))
    print(list(ref))
    print(list(data))
    ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    data.drop(['Unnamed: 0'], inplace=True, axis=1)
    ref.dropna(inplace=True)
    data.dropna(inplace=True)
    print("the number of rows in the original data: ", len(data))
    print("the number of rows in the original reference: ", len(ref))

    return ref, data

def elmo_semantic_similarity():
    """
    Compute semantic similarity using ELMo embeddings
    :return:
    """
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    elmo = ElmoEmbedder(options_file= options_file, weight_file=weight_file)

    ref, data = load_data()
    sim_scores =[]
    for j in range(0, len(data), batch_size):
        batch = data.loc[j:j+batch_size-1]
        start = time.time()

        processed_ref = list(map(lambda x: preprocessing_tweet_text(x), ref['original_text'].values))
        processed_cand = list(map(lambda x: preprocessing_tweet_text(x), batch['text'].values))
        assert len(processed_ref) == len(ref)
        assert len(processed_cand) == len(batch)
        ref['credbank_processed_text'] = processed_ref
        batch['credbank_processed_text'] = processed_cand

        # tokenised_ref = list(map(lambda x: literal_eval(x), ref['credbank_processed_text'].values))
        # tokenised_cand = list(map(lambda x: literal_eval(x), batch['credbank_processed_text'].values))
        tokenised_ref =  list(ref['credbank_processed_text'].values)
        tokenised_cand = list(batch['credbank_processed_text'].values)
        #######################
        # TO DO: Apply bactch #
        #######################
        ## Computes the ELMo embeddings for a batch of tokenized sentences.
        elmo_ref = elmo.embed_batch(tokenised_ref) # a list of tensors
        # print("elmo tweet1 shape ", elmo_tweet1[2].shape)
        elmo_cand = elmo.embed_batch(tokenised_cand)

        ## Compute the mean elmo vector for each tweet
        elmo_ref_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_ref))
        # print("elmo tweet1 avg shape ", elmo_tweet1_avg[0].shape)
        elmo_cand_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_cand))
        elmo_ref_avg = np.squeeze(np.asarray(elmo_ref_avg), axis=1)
        elmo_cand_avg = np.squeeze(np.asarray(elmo_cand_avg), axis=1)
        print("Averaged ELMo vector dimension (ref): ", elmo_ref_avg.shape)
        print("Averaged ELMo vector dimension (cand): ", elmo_cand_avg.shape)
        # assert elmo_ref_avg.shape[0] == elmo_cand_avg.shape[0] == len(batch)

        for i in range(elmo_ref_avg.shape[0]):
            batch_scores = []
            for k in range(elmo_cand_avg.shape[0]):
                print("")
                print(j, i, k)
                score = cosine_similarity(elmo_ref_avg[i,:].reshape(1,-1), elmo_cand_avg[k,:].reshape(1,-1))
                sim_scores.append(score.flatten()[0])
                batch_scores.append(score.flatten()[0])
        end = time.time()
        print("len batch scores ", len(batch_scores))
        print("Time elapsed for bath {}: {}".format(int(j/batch_size), end-start))
        batch['sim_scores'] = batch_scores
        # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
        #                        'data_augmentation/results/ref_batch_{}/{}_batch{}.csv'.format(batch_num, 'elmo_merged', int(j/batch_size)))
        # batch.to_csv(outfile)
    # data['sim_scores'] = sim_scores
    # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/results/elmo/{}.csv'.format('elmo_merged_55b'))
    # data.to_csv(outfile)
    print("Done")

def eval_results():
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/results/elmo/{}.csv'.format('elmo_merged_55b'))
    eval(result_path)

elmo_semantic_similarity()
# eval_results()
import torch
import allennlp
import h5py
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.commands.elmo import ElmoEmbedder
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
import string
from credbankprocessor import preprocessing_tweet_text
from preprocessing import text_preprocessor

from ast import literal_eval # convert string list to actual list
from nltk import TweetTokenizer
from semeval_data_processor import load_csv
from simsem_eval import eval
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from ast import literal_eval

# _stop_words = stopwords.words('english')
# stop_words_filter = lambda t : filter(lambda a: a not in _stop_words, t)
# punctuation_filter = lambda t : filter(lambda a: a not in string.punctuation, t)
#
# lemmatizer = WordNetLemmatizer()
# postag_map = {'j': 'a','n': 'n','v': 'v'}
#
# def _lemmatize(token_list):
#     lemmatized_tokens = [lemmatizer.lemmatize(token, postag_map[tag[0].lower()]) if tag[0].lower() in ['j', 'n', 'v']
#                         else lemmatizer.lemmatize(token) for token, tag in nltk.pos_tag(token_list)]
#     return lemmatized_tokens


pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

event = 'boston'

# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

elmo_credbank_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'rumourdnn', "resource", "embedding",
                                        "elmo_model", "weights_12262018.hdf5")
elmo = ElmoEmbedder(
    # options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
    # weight_file=elmo_credbank_model_path)
    options_file= options_file,
    weight_file= weight_file)


def load_semeval_data():
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

def load_pheme_data():
    ref = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
    cand = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data_augmentation/data/candidates/{}.csv'.format(event))
    # ref = os.path.join(ref_path, '{}_manref_{}.csv'.format(event, batch_num))
    # cand = os.path.join(data_path, '{}.csv'.format(event))

    ref = load_csv(ref)
    data = load_csv(cand)
    print(len(data))
    print(list(ref))
    print(list(data))
    ref = ref[['text']]
    # ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    data.drop(['Unnamed: 0'], inplace=True, axis=1)
    ref.dropna(inplace=True)
    data.dropna(inplace=True)
    print("the number of rows in the original data: ", len(data))
    print("the number of rows in the original reference: ", len(ref))
    ref.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)

    print("the number of rows in the original data: ", len(data))
    print("the number of rows in the original reference: ", len(ref))

    return ref, data


def prepare_input_file():
    # data = load_semeval_data()
    ref, data = load_pheme_data()
    # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input')
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data_augmentation/data/file-embed-input'))
    print(outfile)

    # tokenised_tweet1 = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet1'].values))
    # tokenised_tweet2 = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet2'].values))
    # tokenised_tweet1 = tokenised_tweet1[:100]
    processed_cand = list(map(lambda x: preprocessing_tweet_text(x), data['text'].values))
    processed_ref = list(map(lambda x: preprocessing_tweet_text(x), ref['text'].values))
    tokenised_tweet1 = list(map(lambda x: " ".join(x), processed_cand))
    tokenised_tweet2 = list(map(lambda x: " ".join(x), processed_ref))
    # pp(tokenised_tweet1[:10])
    for t in tokenised_tweet1:
        with open(os.path.join(outfile, 'input-cand.txt'), 'a') as f:
        # with open(os.path.join(outfile, 'input-text1.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
    f.close()

    for t in tokenised_tweet2:
        with open(os.path.join(outfile, 'input-ref.txt'), 'a') as f:
        # with open(os.path.join(outfile, 'input-text2.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
    f.close()

    # with open(os.path.join(outfile, 'input-text2.txt'), 'r') as f:
    #     x = f.readlines()
    # pp(x)

# infile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/data/file-embed-input/input-text1.txt')
# outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/data/file-embed-output/output-text1.hdf5')
# elmo.embed_file(input_file=infile, output_file_path=outfile, output_format='average', forget_sentences=True)
#
# infile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/data/file-embed-input/input-text2.txt')
# outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/data/file-embed-output/output-text2.hdf5')
# elmo.embed_file(input_file=infile, output_file_path=outfile, output_format='average', forget_sentences=True)

prepare_input_file()
#
# f = h5py.File(outfile, 'r')
# print("Keys: %s" % len(f.keys()))
# print(f[('1')].shape)
# f.close()
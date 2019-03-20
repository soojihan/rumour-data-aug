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
from semeval.semeval_data_processor import load_csv
from simsem_eval import eval
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from ast import literal_eval
import argparse
from glob import glob
import re
import random

pd.set_option('display.expand_frame_repr', False)

parser = argparse.ArgumentParser()
parser.add_argument('--event', help='the name of event')
parser.add_argument('--infile_cand', help='path to candidates')
parser.add_argument('--infile_ref', help='path to ref')
parser.add_argument('--score_path', help='path to save scores')
parser.add_argument('--newdf', help='path to save dropped df')
# parser.add_argument('--outfile_cand', help='path to store cand embedding')
# parser.add_argument('--outfile_ref', help='path to store ref embedding')
# print(parser.format_help())

args = parser.parse_args()
event = args.event
infile_cand = args.infile_cand
infile_ref = args.infile_ref
# score_path = args.score_path
# newdf_path = args.newdf
# outfile_cand = args.outfile_cand
# outfile_ref = args.outfile_ref


pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

event = 'charliehebdo'


def load_elmo():
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    #
    elmo_credbank_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'rumourdnn', "resource", "embedding",
                                            "elmo_model", "weights_12262018.hdf5")
    # elmo_credbank_model_path = '/oak01/data/sooji/data-aug/resource/weights_12262018.hdf5'
    elmo = ElmoEmbedder(
        options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
        weight_file=elmo_credbank_model_path)
        # options_file= options_file,
        # weight_file= weight_file)
    return elmo

def load_semeval_data():
    """
    Load semeval-2015 task1 data
    :return:
    """
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
    """
    Load PHEME data or data to be augmented
    :return:
    """
    ref = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
    cand = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data_augmentation/data/candidates/{}.csv'.format(event))
    # ref = os.path.join(ref_path, '{}.csv'.format(event))
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
    """
    1. Generate input files where each line contains a sentence tokenized by whitespace.
    :return: text files (input to ELMo)
    """
    # data = load_semeval_data()
    ref, data = load_pheme_data()
    # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input')
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data_augmentation/data/file-embed-input/{}'.format(event)))
    outfile = os.path.abspath(outfile)
    os.makedirs(outfile, exist_ok=True)
    print(outfile)

    ## Semeval
    # tokenised_tweet1 = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet1'].values))
    # tokenised_tweet2 = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet2'].values))

    ## PHEME / Twitter Event 2012-2016
    processed_cand = list(map(lambda x: preprocessing_tweet_text(x), data['text'].values))
    processed_ref = list(map(lambda x: preprocessing_tweet_text(x), ref['text'].values))
    tokenised_tweet1 = list(map(lambda x: " ".join(x), processed_cand))
    tokenised_tweet2 = list(map(lambda x: " ".join(x), processed_ref))

    for t in tokenised_tweet1:
        # with open(os.path.join(outfile, 'input-cand.txt'), 'a') as f:
        with open(os.path.join(outfile, 'input-text1.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
    f.close()

    for t in tokenised_tweet2:
        # with open(os.path.join(outfile, 'input-ref.txt'), 'a') as f:
        with open(os.path.join(outfile, 'input-text2.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
    f.close()
    print("done")

    # with open(os.path.join(outfile, 'input-text2.txt'), 'r') as f:
    #     x = f.readlines()
    # pp(x)

def remove_empty_lines_from_input(indices):
    """
    Remove empty sentences; ELMo file embed raises error when there're empty strings
    :param indices: indices of empty lines
    :return:
    """
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data_augmentation/data/file-embed-input/{}'.format(event)))

    with open(os.path.join(outfile, 'input-text2.txt'), 'r') as f:
        x = f.read().splitlines()
        lines = [k for k in x if k]
    f.close()
    print(len(lines))
    print(len(indices))
    print(len(x))
    assert len(x) == (len(lines)+len(indices))

    for t in lines:
        with open(os.path.join(outfile, 'input-text2-noempty.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
        f.close()

def eval_results():
    """
    Evaluate results of SemEval task
    :return:
    """
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data/semeval2015/file-embed-output/cred-avg/score.csv'))
    print(result_path)
    eval(result_path)


def data_augmentation(): ### ref0-55 2:49:28am 15 Mar
    """
    Deduplicate the final output (after filtering out by applying a threshold)
    Balance pos and eng examples using the threshold fine tuned usnig the SemEval
    # subset = df[df['sim_score'] >= 0.652584] # 6088
    # subset = df[df['sim_score'] >= 0.691062] # 7000
    # subset = df[df['sim_score'] >= 0.708341] # 7500
    # subset = df[df['sim_score'] >= 0.760198] # 8502
    # subset = df[df['sim_score'] >= 0.801806] # 9003
    # subset = df[df['sim_score'] >= 0.849272] # 0.9506

    :return:
    """
    event = 'boston'
    files = glob(os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/{}/scores/ref*.csv'.format(event)))
    pos_ids = set()
    neg_ids = set()
    # files = files[:4]
    pos_threshold = 0.760198
    precision = 8502
    # neg_threshold = 0.33
    neg_threshold = 0.3
    for f in files: # iter each reference result
        df = pd.read_csv(f)
        pos_subset = df[df['sim_score'] >= pos_threshold]
        sub_pos_ids = list(pos_subset['id'].values)

        neg_subset = df[ df['sim_score'] < neg_threshold]
        # false_neg_subset = df[df['sim_score']>=neg_threshold]
        # false_neg_subset = df[df['sim_score'] <= neg_threshold]

        sub_neg_ids = list(neg_subset['id'].values)
        # false_neg_ids = list(false_neg_subset['id'].values)
        pos_ids.update(sub_pos_ids)
        neg_ids.update(sub_neg_ids)
        print("Updated negative indices ", len(neg_ids))
        # neg_ids -= set(false_neg_ids)
        # print("After removing false indices ", len(neg_ids))
        print("")
    print("*"*10)

    print("Number of positive examples ", len(pos_ids))
    print("Number of negative examples ", len(neg_ids))
    print("")
    num_pos = len(pos_ids)
    infile = '/Users/suzie/Desktop/PhD/Projects/data-aug/data_augmentation/data/file-embed-output/{}/'.format(event)
    full_df = pd.read_csv(os.path.join(infile, 'dropped_df.csv'))
    total_indices = list(full_df.index)
    print("Total number of candidates ", len(total_indices))
    random_neg_indices = random.sample(neg_ids, num_pos*3)
    print(random_neg_indices)
    #
    ## Generate final input df
    full_df.drop(['Unnamed: 0'], inplace=True, axis=1)
    pos_subset = full_df.loc[pos_ids]
    pos_subset['label'] = np.ones(num_pos, dtype=int)
    neg_subset = full_df.loc[random_neg_indices]
    # neg_subset = neg_subset[neg_subset['retweet_count']<100] # filter out negative subset
    # pp(list(neg_subset['text'].values))
    neg_subset['label'] = np.zeros(len(neg_subset), dtype=int)
    result = pd.concat([pos_subset, neg_subset])
    print(len(result))
    save_path = os.path.join(infile, 'results_p{}<0.3'.format(precision))
    os.makedirs(save_path, exist_ok=True)
    result.to_csv(os.path.join(save_path, '{}-{}.csv'.format(event, precision)))

def get_elmo_embeddings():
    """
    Implement ELMo embed file

    :return: hdf5 file (ELMo embeddings per sentence)
    """
    infile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input/input-text1.txt')
    infile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-input/{}/input-text1.txt'.format(event))
    infile = os.path.abspath(infile)
    # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-output/5.5b-avg/output-text1.hdf5')
    outfile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/{}/output-text1.hdf5'.format(event))
    outfile = os.path.abspath(outfile)
    print(outfile)
    elmo.embed_file(input_file=infile, output_file_path=outfile, output_format='average')

    infile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input/input-text2.txt')
    infile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-input/{}/input-text2-noempty.txt'.format(event))

    # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-output/5.5b-avg/output-text2.hdf5')
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/{}/output-text2.hdf5'.format(event)))
    print(outfile)
    elmo.embed_file(input_file=infile, output_file_path=outfile, output_format='average')


    ## andromeda: elmo embeddings
    elmo.embed_file(input_file=infile_cand, output_file_path=outfile_cand, output_format='average')
    elmo.embed_file(input_file=infile_ref, output_file_path=outfile_ref, output_format='average')

def load_empty_indices(event, t='cand'):
    """
    ELMo embed_file raises error if there are empty strings --> remove
    :param: event
    :param: t: 'cand' or 'ref
    :return:
    """
    # boston_empty = [51854, 256585, 296905, 395193, 415348, 417639]
    # ottawa_ref_empty = [3]
    outpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'data_augmentation/data/candidates'))
    # print(outpath)
    # with open(os.path.join(outpath, 'ottawa_ref_empty_index.pickle'), 'wb') as f:
    #     pickle.dump(ottawa_ref_empty, f)
    with open(os.path.join(outpath, '{}_{}_empty_index.pickle'.format(event, t)), 'rb') as f:
        ids = pickle.load(f)
    return ids

# semeval_sem_sim()
# eval_results()
prepare_input_file()
remove_empty_lines_from_input(load_empty_indices(event='orrawa'))
pheme_sem_sim(ottawa_empty, ref_empty=ottawa_ref_empty)
merge_batch_results()
deduplication()

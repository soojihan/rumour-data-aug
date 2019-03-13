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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--event', help='the name of event')
parser.add_argument('--infile_cand', help='path to candidates')
parser.add_argument('--infile_ref', help='path to ref')
parser.add_argument('--outfile_cand', help='path to store cand embedding')
parser.add_argument('--outfile_ref', help='path to store ref embedding')
# print(parser.format_help())

args = parser.parse_args()
event = args.event
infile_cand = args.infile_cand
infile_ref = args.infile_ref
outfile_cand = args.outfile_cand
outfile_ref = args.outfile_ref


pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

# event = 'boston'

# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

# elmo_credbank_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'rumourdnn', "resource", "embedding",
#                                         "elmo_model", "weights_12262018.hdf5")
elmo_credbank_model_path = '/oak01/data/sooji/data-aug/resource/weights_12262018.hdf5'
elmo = ElmoEmbedder(
    options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
    weight_file=elmo_credbank_model_path)
    # options_file= options_file,
    # weight_file= weight_file)
#

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
    """
    Generate input files where each line contains a sentence tokenized by whitespace.
    :return:
    """
    # data = load_semeval_data()
    ref, data = load_pheme_data()
    # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input')
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data_augmentation/data/file-embed-input/{}'.format(event)))
    outfile = os.path.abspath(outfile)
    os.makedirs(outfile, exist_ok=True)
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

    :param indices: indices of empty lines
    :return:
    """
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data_augmentation/data/file-embed-input/{}'.format(event)))

    with open(os.path.join(outfile, 'input-text1.txt'), 'r') as f:
        x = f.read().splitlines()
        lines = [k for k in x if k]
    f.close()
    print(len(lines))
    print(len(indices))
    print(len(x))
    assert len(x) == (len(lines)+len(indices))

    for t in lines:
        with open(os.path.join(outfile, 'input-text1-noempty.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
        f.close()

def semeval_sem_sim():
    text1_emb = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-output/5.5b-avg/output-text1.hdf5')
    text2_emb = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-output/5.5b-avg/output-text2.hdf5')
    f1 = h5py.File(text1_emb, 'r')
    print("Keys: %s" % len(f1.keys()))

    f2 = h5py.File(text2_emb, 'r')
    print("Keys: %s" % len(f2.keys()))
    sim_scores=[]
    tweet_id = []
    gold_labels= []
    pp(f1.keys())
    data = load_semeval_data()
    print(list(data))
    for i, k in enumerate(f1.keys()):
        if i%500==0:
            print(i)

        if k != 'sentence_to_index':
            int_id = int(k)
            label = data.loc[int_id]['goldlabel']
            gold_labels.append(label)
            text1 = np.average(f1[(k)], axis=0).reshape(1,-1)
            text2 = np.average(f2[(k)], axis=0).reshape(1,-1)
            assert text1.shape[1]==text2.shape[1]

            if not (np.isnan(text1).any()) and not (np.isnan(text2).any()):
                score = cosine_similarity(text1, text2)
                sim_scores.append(score.flatten()[0])
                tweet_id.append(k)
                # if score.flatten()[0] >= 0.673580:
                #     print(i, cand_id, score.flatten()[0])
            else:
                score = 0
                sim_scores.append(0)
                tweet_id.append(k)
            # if i==10:
            #     break
    df = pd.DataFrame()
    df['sim_score']=sim_scores
    df['id']=tweet_id
    df['label']=gold_labels
    print(df.head())
    df.sort_values(by=['id'],ascending=True)
    print(df.head())
    outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data/semeval2015/file-embed-output/5.5b-avg'))
    df.to_csv(os.path.join(outfile,'score.csv'))

def eval_results():
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data/semeval2015/file-embed-output/5.5b-avg/score.csv'))
    print(result_path)
    eval(result_path)


#
# # infile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input/input-text1.txt')
# infile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-input/boston/input-text1.txt')
# infile = os.path.abspath(infile)
# # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-output/5.5b-avg/output-text1.hdf5')
# outfile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/boston/output-text1.hdf5')
# outfile = os.path.abspath(outfile)
# print(outfile)
# elmo.embed_file(input_file=infile, output_file_path=outfile, output_format='average')
#
# # infile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input/input-text2.txt')
# infile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-input/boston/input-text2.txt')
#
# # outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-output/5.5b-avg/output-text2.hdf5')
# outfile = os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/boston/output-text2.hdf5')
#
# elmo.embed_file(input_file=infile, output_file_path=outfile, output_format='average')


## andromeda
elmo.embed_file(input_file=infile_cand, output_file_path=outfile_cand, output_format='average')
elmo.embed_file(input_file=infile_ref, output_file_path=outfile_ref, output_format='average')

#######################################################################################################################################
# semeval_sem_sim()
# eval_results()
# prepare_input_file()
# remove_empty_lines_from_input(boston_empty)
# f = h5py.File(outfile, 'r')
# print("Keys: %s" % len(f.keys()))
# print(f[('1')].shape)
# f.close()

#### TODO: boston missing indices --->
# boston_empty = [51854, 256585, 296905, 395193, 415348, 417639]

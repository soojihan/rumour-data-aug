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
parser.add_argument('--empty_path', help='path to the indices of empty strings')
# parser.add_argument('--outfile_ref', help='path to store ref embedding')
# print(parser.format_help())

args = parser.parse_args()
event = args.event
infile_cand = args.infile_cand
infile_ref = args.infile_ref
score_path = args.score_path
newdf_path = args.newdf
empty_indice_path = args.empty_path
# outfile_cand = args.outfile_cand
# outfile_ref = args.outfile_ref


pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

# event = 'ottawa'

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
    # outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data/semeval2015/file-embed-output/5.5b-avg'))
    # df.to_csv(os.path.join(outfile,'score.csv'))

def pheme_sem_sim(cand_empty, ref_empty=None ):

    ## Load ELMo embedding dictionaries
    cand_emb = infile_cand
    ref_emb = infile_ref
    f1 = h5py.File(cand_emb, 'r')
    print("Keys: %s" % len(f1.keys()))

    f2 = h5py.File(ref_emb, 'r')
    print("Keys: %s" % len(f2.keys()))

    ## Load candidates and references
    ref, data = load_pheme_data()
    data = data.drop(cand_empty)
    # data.reset_index(inplace=True, drop=True)
    if not ref_empty is None: ## Drop empty references
        ref = ref.drop(ref_empty)
        ref.reset_index(inplace=True, drop=True)
        ref.to_csv(os.path.join(newdf_path, 'dropped_ref.csv'))
    data.to_csv(os.path.join(newdf_path, 'dropped_df.csv')) # Save the dropped dataframe
    assert len(data) == (len(f1.keys())-1)
    print(list(data))

    for i, ref_k in enumerate(f2.keys()): # Iterate reference embeddings
        sim_scores = []
        tweet_id = []
        print(i)
        for j, cand_k in enumerate(f1.keys()):
            if j % 1000 ==0:
                print(j)
            if (ref_k != 'sentence_to_index') and (cand_k!='sentence_to_index'):

                ref_id = int(ref_k)
                cand_id = int(cand_k)
                text1 = np.average(f1[(cand_k)], axis=0).reshape(1, -1)
                text2 = np.average(f2[(ref_k)], axis=0).reshape(1, -1)
                assert text1.shape[1] == text2.shape[1]

                if not (np.isnan(text1).any()) and not (np.isnan(text2).any()):
                    score = cosine_similarity(text1, text2)
                    sim_scores.append(score.flatten()[0])
                    tweet_id.append(cand_id)
                    # if score.flatten()[0] >= 0.673580:
                    #     print(i, cand_id, score.flatten()[0])
                else:
                    score = 0
                    sim_scores.append(0)
                    tweet_id.append(cand_id)
                # if i==10:
                #     break
        df = pd.DataFrame()
        df['sim_score'] = sim_scores
        df['id'] = tweet_id
        print(df.head())
        df.sort_values(by=['id'], ascending=True)
        print(df.head())
        # outfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data/semeval2015/file-embed-output/5.5b-avg'))
        df.to_csv(os.path.join(score_path,'ref{}_score.csv'.format(ref_k))) # Save scores per reference tweet
#
def eval_results():
    """
    Evaluate results of SemEval task
    :return:
    """
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'data/semeval2015/file-embed-output/cred-avg/score.csv'))
    print(result_path)
    eval(result_path)

# empty_indices_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data_augmentation/data/candidates'))
# newdf_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data_augmentation/data/file-embed-output/ottawa'))
# print(empty_indices_path)
with open(os.path.join(empty_indice_path), 'rb') as f:
    empty_indices = pickle.load(f)

pheme_sem_sim(empty_indices)


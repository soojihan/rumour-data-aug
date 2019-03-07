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
import nltk
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
from nltk.corpus import stopwords
import string
from nltk import WordNetLemmatizer
from glob import glob
import argparse


# _stop_words = stopwords.words('english')
# _stop_words.extend(['bostonbombings', 'boston', 'pray', 'prayforboston', 'marathon'])
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
# # lemmatize_tokens = lambda t: list(map(lambda a: lemmatizer.lemmatize(a, 'v'), t))

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
parser = argparse.ArgumentParser()
parser.add_argument('--batch_num', help='the index of the batch')
parser.add_argument('--save_dir', help='path to save the results')
parser.add_argument('--ref_path', help='path to reference')
parser.add_argument('--cand_path', help='path to candidates')
# print(parser.format_help())
args = parser.parse_args()
batch_num = args.batch_num
save_dir = args.save_dir
ref_path = args.ref_path
data_path = args.cand_path
print("save_dir ", save_dir)
batch_size = 1000

print("Hello World")

def load_data():
    # ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/data/ref/boston_manref_{}.csv'.format(batch_num))
    # cand = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                          'data_augmentation/data/candidates/boston_candidates_rts_more-than-4-tokens.csv')
    ref = os.path.join(ref_path, 'boston_manref_{}.csv'.format(batch_num))
    cand = os.path.join(data_path, 'boston_candidates_rts_more-than-4-tokens.csv')

    ref = load_csv(ref)
    data = load_csv(cand)
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
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    #
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    elmo = ElmoEmbedder(options_file= options_file, weight_file=weight_file)

    ref, data = load_data()
    processed_ref = list(map(lambda x: preprocessing_tweet_text(x), ref['text'].values))
    # processed_ref = list(map(lambda x: list(stop_words_filter(x)), processed_ref))
    # processed_ref = list(map(lambda x: list(punctuation_filter(x)), processed_ref))
    # processed_ref = list(map(lambda x: _lemmatize(x), processed_ref))
    # processed_ref = list(map(lambda x: list(lemmatize_tokens(x)), processed_ref))
    assert len(processed_ref) == len(ref)
    ref['credbank_processed_text'] = processed_ref
    tokenised_ref = list(ref['credbank_processed_text'].values)
    elmo_ref = elmo.embed_batch(tokenised_ref)  # a list of tensors
    elmo_ref_avg = list(map(lambda x: np.mean(x[2], axis=0).reshape(1, -1), elmo_ref))
    elmo_ref_avg = np.squeeze(np.asarray(elmo_ref_avg), axis=1)
    print("Averaged ELMo vector dimension (ref): ", elmo_ref_avg.shape)

    for i in range(elmo_ref_avg.shape[0]): # iter ref tweet
        start_ref = time.time()
        sim_scores =[]
        for j in range(0, len(data), batch_size): # iter candidate set
            # if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)),
            #                        'data_augmentation/results/ref_batch_{}/elmo_ref{}_batch{}.csv'.format(batch_num, i, int(j / batch_size)))):
            if os.path.exists(os.path.join(save_dir, 'ref_batch_{}/elmo_ref{}_batch{}.csv'.format(batch_num, i, int(j / batch_size)))):
                print("exists")
                continue

            print("ref {} candidate batch {}/{}  ".format(i, j, len(data)))
            batch = data.loc[j:j+batch_size-1]
            start = time.time()
            processed_cand = list(map(lambda x: preprocessing_tweet_text(x), batch['text'].values))
            # processed_cand = list(map(lambda x: stop_words_filter(x), batch['text'].values))
            # processed_cand = list(map(lambda x: punctuation_filter(x), batch['text'].values))
            # processed_cand = list(map(lambda x: _lemmatize(x), processed_ref))
            assert len(processed_cand) == len(batch)
            batch['credbank_processed_text'] = processed_cand
            tokenised_cand = list(batch['credbank_processed_text'].values)
            ## Computes the ELMo embeddings for a batch of tokenized sentences.
            elmo_cand = elmo.embed_batch(tokenised_cand)

            ## Compute the mean elmo vector for each tweet
            elmo_cand_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_cand))
            elmo_cand_avg = np.squeeze(np.asarray(elmo_cand_avg), axis=1)
            print("Averaged ELMo vector dimension (cand): ", elmo_cand_avg.shape)
            assert elmo_cand_avg.shape[0] == len(batch)

            print("NaN ", np.where(np.isnan(elmo_cand_avg)))
            batch_scores = []
            for k in range(elmo_cand_avg.shape[0]): # iter each batch
                if not np.isnan(elmo_cand_avg[k,:].reshape(1,-1)).any():
                    score = cosine_similarity(elmo_ref_avg[i,:].reshape(1,-1), elmo_cand_avg[k,:].reshape(1,-1))
                    sim_scores.append(score.flatten()[0])
                    batch_scores.append(score.flatten()[0])
                else:
                    score=0
                    sim_scores.append(0)
                    batch_scores.append(0)
            end = time.time()
            # print("len batch scores ", len(batch_scores))
            print("Time elapsed for batch {}: {}".format(int(j/batch_size), end-start))
            print("")
            batch['sim_scores'] = batch_scores
            # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
            #                        'data_augmentation/results/ref_batch_{}'.format(batch_num))
            outfile = os.path.join(save_dir, 'ref_batch_{}'.format(batch_num))
            os.makedirs(outfile, exist_ok=True)
            batch.to_csv(os.path.join(outfile, 'elmo_ref{}_batch{}.csv'.format(i, int(j/batch_size))))

        print("data length: ", len(data), "sim scores len: ", len(sim_scores))
        if len(sim_scores) >0:
            data['sim_scores'] = sim_scores
            # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/results/ref_batch_{}'.format(batch_num))
            outfile = os.path.join(save_dir, 'ref_batch_{}'.format(batch_num))
            data.to_csv(os.path.join(outfile, 'elmo_merged_55b_ref{}.csv'.format(i)))
        end_ref = time.time()
        print("")
        print("---- Time elapsed for ref {}: {} minutes ".format(i, (end_ref-start_ref)/60))
    print("Done")

def eval_results():
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/results/ref_batch_2/elmo_merged_55b_ref0.csv')
    eval(result_path)

def merge_batch_results():
    batch_n = 2
    ref_n = 1
    merged_df = pd.DataFrame()
    files = glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/results/ref_batch_{}/elmo_ref{}*.csv'.format(batch_n, ref_n)))
    pp(files)
    for x in files:
        df = pd.read_csv(x)
        merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)

        # print(merged_df)
    print(len(merged_df))
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'data_augmentation/results/ref_batch_{}'.format(batch_n))
    merged_df.to_csv(os.path.join(outfile, 'elmo_merged_55b_ref{}.csv'.format(ref_n)))

    subset = merged_df[merged_df['sim_scores'] >= 0.673580]
    subset.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'data_augmentation/results/ref_batch_{}/ref{}_subset.csv'.format(batch_n, ref_n)))
    print(df.head())

#
elmo_semantic_similarity()
# eval_results()
# merge_batch_results()



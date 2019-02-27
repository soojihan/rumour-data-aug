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

def load_csv(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)

    return df

def read_test_label(testlabelfile):
    """
    Read test.label file
    :return: list of gold labels and scores
    """
    # read in golden labels
    goldlabels = []
    goldscores = []

    hasscore = False
    with open(testlabelfile) as tf:
        for tline in tf:
            tline = tline.strip()
            tcols = tline.split('\t')
            if len(tcols) == 2:
                goldscores.append(float(tcols[1]))
                if tcols[0] == "true":
                    goldlabels.append(True)
                elif tcols[0] == "false":
                    goldlabels.append(False)
                else:
                    goldlabels.append(None)
    return goldlabels, goldscores

def prepare_test_data():
    """
    Remove retweets from the original data in order to find source tweets from Tweet Event 2012-2016 dataset
    :return: csv
    """
    testlabelfile=  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data/test.label')
    goldlabels, goldscores = read_test_label(testlabelfile)

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data')
    with open(os.path.join(data_path, 'test.data'), 'r') as f:
        test_data = f.read().splitlines()
        test_data = [x.split("\t") for x in test_data]
    print(len(test_data))

    df = pd.DataFrame(columns=['topic', 'tweet1', 'tweet2'])
    for i, sample in enumerate(test_data):
        df.loc[len(df)] = sample[1:4]

    # processed_tweet1 = list(map(lambda x: text_preprocessor(x, ignore_retweet=False), df['tweet1'].values))
    processed_tweet1 = list(map(lambda x: preprocessing_tweet_text(x), df['tweet1'].values))
    processed_tweet2 = list(map(lambda x: preprocessing_tweet_text(x), df['tweet2'].values))
    df['goldlabel'] = goldlabels
    df['goldscore'] = goldscores

    df['processed_tweet1'] = processed_tweet1
    df['processed_tweet2'] = processed_tweet2
    print(goldlabels)
    print([type(x) for x in goldlabels])
    df['goldlabel'].replace([False, True], [0,1], inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    print(df)
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data/{}.csv'.format('test'))
    df.to_csv(outfile)
    print("Done")

def prepare_train_dev_data():
    """
    Convert trian.data and dev.data to .csv
    Add processed tweets and binary labels
    :return:
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data')
    with open(os.path.join(data_path, 'train.data'), 'r') as f:
        train_data = f.read().splitlines()
        train_data = [x.split("\t") for x in train_data]
    print(len(train_data))
    # train_data = train_data[:4]
    df = pd.DataFrame(columns=['topic', 'tweet1', 'tweet2', 'label'])
    for i, sample in enumerate(train_data):
        df.loc[len(df)] = sample[1:5]
    # processed_tweet1 = list(map(lambda x: text_preprocessor(x, ignore_retweet=False), df['tweet1'].values))
    processed_tweet1 = list(map(lambda x: preprocessing_tweet_text(x), df['tweet1'].values))
    processed_tweet2 = list(map(lambda x: preprocessing_tweet_text(x), df['tweet2'].values))

    df['goldlabel'] = df['label'] # copy label column
    df['goldlabel'].replace(['(3, 2)', '(4, 1)', '(5, 0)'], 1, inplace=True)
    df['goldlabel'].replace(['(1, 4)', '(0, 5)'], 0, inplace=True)
    df['goldscore'] = df['label'].copy()
    df.drop(['label'], axis=1, inplace=True)
    print(list(df))
    print(df.head())

    df['processed_tweet1'] = processed_tweet1
    df['processed_tweet2'] = processed_tweet2
    df.drop(df[df.goldscore== '(2, 3)'].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data/{}.csv'.format('train'))
    df.to_csv(outfile)
    print("Done")

def merge_train_dev_test():
    target_file_names = ['test', 'dev', 'train']
    merged_df = pd.DataFrame(columns= ['topic', 'tweet1', 'tweet2','goldlabel', 'goldscore', 'processed_tweet1', 'processed_tweet2' ])
    for d_type in target_file_names:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data/{}.csv'.format(d_type))
        print("data path ", data_path)
        data = load_csv(data_path)
        print(len(data))
        print(list(data))
        data.drop(['Unnamed: 0'], inplace=True, axis=1)
        print(data.isnull().any().any())
        if data.isnull().any().any():
            raise ValueError
        print("the number of rows in the {} data: ".format(d_type), len(data))
        merged_df = merged_df.append(data, ignore_index=True)
    print(merged_df)
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/data/merged_semeval.csv'.format('train'))
    merged_df.to_csv(outfile)
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
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

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
        outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'data/semeval2015/{}_batch{}.csv'.format('{}_elmo_merged'.format(data_type), int(j/batch_size)))
        batch.to_csv(outfile)
    data['sim_scores'] = sim_scores
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/{}.csv'.format('{}_elmo_merged'.format(data_type)))
    data.to_csv(outfile)
    print("Done")

def load_pretrained_glove():
    embeddings_index = dict()
    glove_path = os.path.join(os.path.dirname(__file__), '..', 'data/glove.twitter.27B/glove.twitter.27B.25d.txt')
    glove_abs_path = os.path.abspath(glove_path)

    with open(glove_abs_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    # print(embeddings_index)
    outfile = os.path.join(os.path.dirname(__file__), '..', '..','..','Data','glove.twitter.27B')
    outfile_abs_path =os.path.abspath(outfile)
    with open(os.path.join(outfile_abs_path, 'glove_25d_dict.pickle'), 'wb') as f:
        pickle.dump(embeddings_index, f)

    # with open(os.path.join(outfile_abs_path, 'glove_200d_dict.pickle'), 'rb') as f:
    #     x = pickle.load(f)
    # print(x)
    return embeddings_index

def glove_semantic_similarity():
    glove_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'glove.twitter.27B')
    glove_abs_path = os.path.abspath(glove_path)
    with open(os.path.join(glove_abs_path, 'glove_25d_dict.pickle'), 'rb') as f:
        glove_embeddings = pickle.load(f)

    data = load_data()

    sim_scores = []
    for j in range(0, len(data), batch_size):
        batch = data.loc[j:j + batch_size - 1]
        start = time.time()

        tokenised_tweet1 = list(map(lambda x: literal_eval(x), batch['processed_tweet1'].values))
        tokenised_tweet2 = list(map(lambda x: literal_eval(x), batch['processed_tweet2'].values))
        batch_embeddings = np.empty((0, 25))
        ## Computes the ELMo embeddings for a batch of tokenized sentences.
        for tweet in tokenised_tweet1:
            print(type(tweet))
            tweet_embeddings = np.empty((0, 25))
            for token in tweet:
                print(token)
                if token in glove_embeddings:
                    print(type(glove_embeddings[token]))
                    tweet_embeddings = np.mean(np.vstack((tweet_embeddings,glove_embeddings[token])), axis=0)
                    print(tweet_embeddings.shape)
                    tweet_embeddings = tweet_embeddings.reshape(1,-1)
                    print(tweet_embeddings.shape)
                    batch_embeddings = np.vstack(tweet_embeddings)
        print(batch_embeddings.shape)



        raise SystemExit
        ## Compute the mean elmo vector for each tweet
        elmo_tweet1_avg = list(map(lambda x: np.mean(x[2], axis=0).reshape(1, -1), elmo_tweet1))
        elmo_tweet2_avg = list(map(lambda x: np.mean(x[2], axis=0).reshape(1, -1), elmo_tweet2))
        elmo_tweet1_avg = np.squeeze(np.asarray(elmo_tweet1_avg), axis=1)
        elmo_tweet2_avg = np.squeeze(np.asarray(elmo_tweet2_avg), axis=1)
        print("Averaged ELMo vector dimension (tweet1): ", elmo_tweet1_avg.shape)
        print("Averaged ELMo vector dimension (tweet2): ", elmo_tweet2_avg.shape)
        assert elmo_tweet1_avg.shape[0] == elmo_tweet2_avg.shape[0] == len(batch)

        batch_scores = []
        for i in range(elmo_tweet1_avg.shape[0]):
            score = cosine_similarity(elmo_tweet1_avg[i, :].reshape(1, -1), elmo_tweet2_avg[i, :].reshape(1, -1))
            # sim_scores[i] = score.flatten()[0]
            sim_scores.append(score.flatten()[0])
            batch_scores.append(score.flatten()[0])
        end = time.time()
        print("Time elapsed for bath {}: {}".format(int(j / batch_size), end - start))
        batch['sim_scores'] = batch_scores
        # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
        #                        'data/semeval2015/{}_batch{}.csv'.format('{}_elmo_merged'.format(data_type),
        #                                                                 int(j / batch_size)))
        # batch.to_csv(outfile)
    data['sim_scores'] = sim_scores
    # outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                        'data/semeval2015/{}.csv'.format('{}_elmo_merged'.format(data_type)))
    # data.to_csv(outfile)
    print("Done")

def elmo_test():
    """
    Find the best threshold which achieves the maximum F-measure
    :return:
    """
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/semeval2015/{}.csv'.format('merged_semeval_elmo'))
    with open(result_path, 'r') as f:
        df = pd.read_csv(f)
        print(df.head())
    print(len(df))
    df.rename(columns={'binary_label': 'goldlabel'}, inplace=True)
    df[['goldlabel']] = df[['goldlabel']].astype(int)
    # df.sort_values(by='sim_scores', ascending=True, inplace=True)
    threshold_candidates = list(set(df['sim_scores'].values))
    threshold_candidates = [x for x in threshold_candidates if x>=0.5]
    print(len(threshold_candidates))
    print(df.head())
    max_F = 0
    max_P = 0
    max_R = 0
    optimum_threshold = 0
    for threshold in threshold_candidates:
        # print("Threshold ", threshold)
        df.loc[df[df.sim_scores >= threshold].index, 'syslabel'] = 1
        df.loc[df[df.sim_scores < threshold].index, 'syslabel'] = 0
        syslabels = df['syslabel'].values
        goldlabels = df['goldlabel'].values
        F, P, R = eval(syslabels, goldlabels)
        if max_F < F:
            max_F = F
            max_P = P
            max_R = R
            optimum_threshold = threshold
            print("max F-measure: {:0.4f}, P: {:0.4f}, R: {:0.4f}, threshold: {:0.6f}".format(max_F, max_P, max_R, optimum_threshold))

    raise SystemExit


def test():
    glove_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'glove.twitter.27B')
    glove_abs_path = os.path.abspath(glove_path)
    with open(os.path.join(glove_abs_path, 'glove_25d_dict.pickle'), 'rb') as f:
        glove_embeddings = pickle.load(f)


    tweet_1 = "RT @TheManilaTimes: Cheers, tears welcome Pope Francis - The Manila Times OnlineThe Manila Times Online http://www.manilatimes.net/cheers-tears-welcome-pope-francis/155612/ …	3:31 am - 15 Jan 2015"
    pre_tweet_1 = preprocessing_tweet_text(tweet_1)
    print(pre_tweet_1)

    tweet_2 = "Welcome to the Philippines Pope Francis @Pontifex Pray for the Philippines & the entire world.	5:19 pm - 15 Jan 2015"
    pre_tweet_2 = preprocessing_tweet_text(tweet_2)
    print(pre_tweet_2)

    tweet_3 = "Retweet if you're proud Filipino! \"Welcome to the Philippines Pope Francis\" http://bit.ly/150Zqcq  ۞| http://bit.ly/1INBcie 	3:02 pm - 15 Jan 2015"
    pre_tweet_3 = preprocessing_tweet_text(tweet_3)
    print(pre_tweet_3)

    tweet_4 = "Why Lambert, Lovren and Lallana have struggled at Liverpool http://dlvr.it/8gqKRv  @PLNewsNow"
    pre_tweet_4 = preprocessing_tweet_text(tweet_4)
    print(pre_tweet_4)


def eval(syslabels, goldlabels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for (i, syslabel) in enumerate(syslabels):

        # if syslabel == True and goldlabels[i] == True:
        if syslabel == 1 and goldlabels[i] == 1:
            tp += 1
        elif syslabel == 1 and goldlabels[i] == 0:
            fp += 1
        elif syslabel == 0 and goldlabels[i] == 0:
            tn += 1
        elif syslabel == 0 and goldlabels[i] == 1:
            fn += 1

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)

    testsize = str(tp + fn + fp + tn)
    return F, P, R

# prepare_test_data()
# read_test_label()
# prepare_train_dev_data()
# semeval_test()
# elmo_semantic_similarity()
glove_semantic_similarity()
# elmo_test()
# merge_train_dev_test()
# load_pretrained_glove()
# test()
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




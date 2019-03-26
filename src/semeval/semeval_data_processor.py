import scipy as sp
import numpy as np
import time
import os
import pandas as pd
import sys
import json
import pickle
from pprint import pprint as pp

import operator
import csv
# from textcleaner import tokenize_by_word
from src.credbankprocessor import preprocessing_tweet_text
from preprocessing import text_preprocessor

from ast import literal_eval # convert string list to actual list


"""

"""
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

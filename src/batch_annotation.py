import os
import json
import pandas as pd
import numpy as np
from pprint import pprint as pp
from textcleaner import tokenize_by_word
import re
import spacy
import sys
import time

import torch

nlp = spacy.load('en_core_web_lg')

## Loading GloVe vectors
# nlp = spacy.load('en')
# glove_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data')
# print(os.path.abspath(glove_path))
# nlp.vocab.vectors.from_glove(os.path.join(os.path.abspath(glove_path), 'glove.twitter.27B'))
pd.set_option('display.max_columns', None)
# n_vectors = 105000
# removed_words = nlp.vocab.prune_vectors(n_vectors)

# doc1 = nlp('break bomb report near finish line add story develop')
# # doc2 = nlp('break authority investigate repo two explosion finish line')
# doc2 = nlp('break report several people injured explosion finish line')
# print(doc1.similarity(doc2))

# batch_num = sys.argv[1]
batch_num = 8

def load_csv(file_path):
    with open(file_path, 'rb') as f:
        df = pd.read_csv(f, encoding = "utf-8")

    return df

def segment_csv():
    """
    Divide reference tweets into multiple batches
    :return:
    """
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_augmentation/data/ref/man_ref.csv')
    ref = load_csv(ref_path)
    # ref.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
    # ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    # ref = ref[2:]
    ref.reset_index(inplace=True, drop=True)
    ref = ref[['text']]
    print(len(ref))
    print(ref.head(2))
    print(ref.tail(2))
    num_segmentations = list(np.arange(10)*10)
    print(num_segmentations)

    for i, num in enumerate(num_segmentations):
        if i>0:
            print(num_segmentations[i-1] ,num)
            subset = ref.loc[num_segmentations[i-1]: min(num, len(ref))-1]
            subset.reset_index(inplace=True, drop=True)
            print(subset)
            out = os.path.join('..', 'data_augmentation/data/ref')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, '{}_manref_{}.csv'.format("boston", i)), 'w') as f:
                subset.to_csv(f)
        else:
            continue

def semantic_similarity():
    # ref_path = os.path.join('..', 'bostonbombings_reference.csv')
    # ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_reference.csv')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_augmentation/data/boston_{}.csv'.format(batch_num))

    # data_path = os.path.join('..','bostonbombings_candidates.csv')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_candidates_with_rts.csv')

    ref = load_csv(ref_path)
    data = load_csv(data_path)
    print(len(data))
    ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    data.drop(['Unnamed: 0', 'Unnamed: 0.1'], inplace=True, axis=1)
    ref.dropna(inplace=True)
    data.dropna(inplace=True)
    print("the number of rows in the original data: ", len(data))
    print("the number of rows in the original reference: ", len(ref))

    data.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
    ref.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
    data.reset_index(inplace=True, drop=True)
    ref.reset_index(inplace=True, drop=True)
    print(list(ref))
    print(list(data))


    print("the number of rows in the deduplicated data: ", len(data))
    print("the number of rows in the deduplicated reference: ", len(ref))
    result_columns = ['ref_original', 'ref_processed', 'sim_score', 'status', 'id',
                      'screen_name', 'text', 'created_at', 'retweet_count', 'processed_text']
    result_df = pd.DataFrame(columns=result_columns)
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'train_augmentation/sim_scores')
    os.makedirs(result_path, exist_ok=True)
    pp(data)

    for i, ref_row in ref.iterrows():
        # ref_tweet = nlp(ref_row['original_text'])
        ref_tweet = ref_row['processed_text']
        print(ref_tweet)
        ref_tweet = nlp(ref_tweet)
        start = time.time()
        for j, data_row in data.iterrows():
            print(i, j, "/" ,len(data))
            # print(list(data_row))
            tweet = data_row['processed_text']
            # print(tweet)
            tweet = nlp(tweet)
            sim_score = ref_tweet.similarity(tweet)
            end= time.time()
            print(end-start)
            if sim_score > 0.5:
                # print("** ref: ", ref_tweet)
                # print("** tweet: ", tweet)
                # print(sim_score)
                result_df.loc[len(result_df)] = [ref_row['original_text'], ref_row['processed_text'], sim_score]+list(data_row)
            # print(result_df)
            # print("")
            # if j==10:
            #     break
        # if i==1:
        #     break
            #
            # result_df.to_csv(os.path.join(result_path, 'boton_{}.csv'.format(batch_num)))
        # print("...Reference {} is complete".format(i))

# semantic_similarity()
segment_csv()


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
import string
# from credbankprocessor import preprocessing_tweet_text
from src.credbankprocessor import preprocessing_tweet_text
from ast import literal_eval # convert string list to actual list
import argparse
from optparse import OptionParser
from glob import glob
import re
import random
import jsonlines
from sys import platform

"""
Convert Twitter events 2012-2016 dataset downloaded using Hydrator into DATAFRAME which comprises of English tweets
Preliminary step for generating input to ELMo embed_file method (https://github.com/allenai/allennlp/blob/master/allennlp/commands/elmo.py)
"""

pd.set_option('display.expand_frame_repr', False)
def load_abspath(x):
    return os.path.abspath(x)

def convert_hydrator_data(event, idfile, tweets):
    """
    1. Convert data obtained using Hydrator (.jsonl) to dataframe (.pickle)
    https://github.com/DocNow/hydrator
    :param event: event name
    :param idfile: filename containing tweet ids
    :param tweets: filename containint downloaded tweets
    :return:
    """
    global temp_path
    if platform == 'darwin':
        ## Path to Tweet objects (.jsonl)
        data_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/downloaded_data/hydrator'))

        ## Path to save the output
        temp_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/saved_data/source-tweets/{}'.format(event)))
        os.makedirs(temp_path, exist_ok=True)

        ## Path to Tweet ids (e.g., Twitter events 2012-2016 ids)
        id_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/downloaded_data'))

    elif platform=='linux':
        ## Path to Tweet objects (.jsonl)
        data_path = load_abspath('/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/data_hydrator/downloaded_data/hydrator')

        ## Path to save the output
        temp_path = load_abspath('/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/data_hydrator/saved_data/source-tweets/{}'.format(event))
        os.makedirs(temp_path, exist_ok=True)

        ## Path to Tweet ids (e.g., Twitter events 2012-2016 ids)
        id_path = load_abspath('/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/data_hydrator/downloaded_data')
    else:
        print("The current platform is not supported...")
        raise ValueError

    ## Load ids
    with open(os.path.join(id_path, idfile), 'r') as f:
        ids = f.read().splitlines()
        f.close()
    processed_ids = []

    df = pd.DataFrame(columns=['id', 'created_at','text', 'processed_text'], dtype=str)
    with jsonlines.open(os.path.join(data_path, tweets)) as reader: # load tweets obtained uinsg Hydrator (.jsonl)
        for i, obj in enumerate(reader):
            source_id = obj['id_str']
            text = obj['full_text']
            created_at = obj['created_at']
            # if source_id == '580319845394325504':
            #     print(obj)

            assert source_id in ids

            # Filter out non-english tweets
            if obj['lang'] == 'en':
                processed_ids.append(source_id)
                processed_text = " ".join(preprocessing_tweet_text(text))
                df.loc[len(df)] = [source_id, created_at, text, processed_text]
                # print(df)
                # print(df.loc[len(df)-1 ,'id'])
                assert df.loc[len(df)-1 ,'id'] == source_id
                # df.to_csv(os.path.join(temp_path, 'input-cand.csv'))
                if i % 1000==0:
                    print(i)
                assert len(df) == len(processed_ids)
                with open(os.path.join(temp_path, 'input-cand.pickle'), 'wb') as tf:
                    pickle.dump(df, tf)
                # with open(os.path.join(temp_path, 'input-text_cand.txt'), 'a') as tf:
                #     tf.write(processed_text)
                #     tf.write("\n")

                with open(os.path.join(temp_path, 'eng_cand_ids.txt'), 'a') as f:
                    f.write(source_id)
                    f.write("\n")

    reader.close()
        # tf.close()
        # f.close()

def main():
    if platform == 'linux':
        parser = OptionParser()
        parser.add_option(
            '--event', dest='event', default='germanwings',
            help='The name of event: default=%default')
        parser.add_option(
            '--idname', dest='idname', default='2015-germanwings-crash.ids',
            help='The name of the file containing Tweet ids: default=%default')
        parser.add_option(
            '--tweets', dest='tweets', default='germanwings.jsonl',
            help='The name of the file containing hydrated Tweets (.jsonl): default=%default')

        (options, args) = parser.parse_args()
        event = options.event
        idname = options.idname
        tweets = options.tweets
    elif platform == 'darwin':
        event ='germanwings'
        idname = '2015-germanwings-crash.ids'
        tweets = 'germanwings.jsonl'

    convert_hydrator_data(event, idname, tweets)

main()

# print("-------------")
# with open(os.path.join(temp_path, 'eng_cand_ids.txt'), 'r') as f:
#     x = f.read().splitlines()
#     pp(x)
# df = pd.read_csv(os.path.join(temp_path, 'input-cand.csv'), dtype=str)
# pp(list(df['id'].values))
# print("-----------")
# assert len(x) == len(df['id'].values)
# for i, x in zip(x, list(df['id'].values)):
#     if i!=x:
#         print(i, x)
#

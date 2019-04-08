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
from glob import glob
import re
import random
import jsonlines
from sys import platform
import datetime
"""
Convert Twitter events 2012-2016 dataset downloaded using Hydrator into DATAFRAME which comprises of English tweets
Preliminary step for generating input to ELMo embed_file method (https://github.com/allenai/allennlp/blob/master/allennlp/commands/elmo.py)
"""

pd.set_option('display.expand_frame_repr', False)
def load_abspath(x):
    return os.path.abspath(x)

def convert_downloaded_data(event, tweets):
    """
    1. Convert data obtained using Hydrator/Twint (.jsonl) to dataframe (.pickle)
    https://github.com/DocNow/hydrator
    :param event: event name
    :param tweets: filename containint downloaded tweets
    :return: dataframe (input-cand-user.pickle)
    """
    global temp_path
    if platform == 'darwin':
        ## Path to Tweet objects (.jsonl)
        data_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/downloaded_data/hydrator'))

        ## Path to save the output
        temp_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/saved_data/source-tweets/{}'.format(event)))
        os.makedirs(temp_path, exist_ok=True)

    elif platform=='linux':
        ## Path to Tweet objects (.jsonl)
        data_path = load_abspath('/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/data_hydrator/downloaded_data/hydrator')

        ## Path to save the output
        temp_path = load_abspath('/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/data_hydrator/saved_data/source-tweets/{}'.format(event))
        os.makedirs(temp_path, exist_ok=True)

    else:
        print("The current platform is not supported...")
        raise ValueError

    processed_ids = []
    texts = []
    dates = []
    lang =[]
    user_ids = []
    user_names = []
    # df = pd.DataFrame(columns=['id', 'created_at','text', 'processed_text'], dtype=str)
    # df = pd.DataFrame(columns=['id', 'created_at','text', 'lang'], dtype=str)
    df = pd.DataFrame(columns=['id', 'created_at','text'], dtype=str)
    with jsonlines.open(os.path.join(data_path, tweets)) as reader: # load tweets obtained uinsg Hydrator (.jsonl)
        start = time.time()
        for i, obj in enumerate(reader):

            ## Hydrator objects (pheme, bostonbombings)
            source_id = obj['id_str']
            text = obj['full_text']
            lang.append(obj['lang'])
            screen_name = obj['user']['screen_name']
            user_id = obj['user']['id_str']
            created_at = obj['created_at']

            ## Twint objects (manchesterbombings, christchurch)
            # source_id = str(obj['id'])
            # text = obj['tweet']
            # created_at = obj['created_at']
            # created_at =  datetime.datetime.fromtimestamp(obj['created_at'] / 1e3)
            # screen_name = obj['username']
            # user_id = str(obj['user_id'])
            # lang.append('en')

            processed_ids.append(source_id)
            texts.append(text)
            dates.append(created_at)
            user_ids.append(user_id)
            user_names.append(screen_name)
            if i % 5000 ==0:
                print(i)
    reader.close()
    df['id'] = processed_ids
    df['created_at'] = dates
    df['text'] = texts
    df['lang'] = lang
    df['screen_name'] = user_names
    df['user_id'] = user_ids
    df = df[df.lang=='en']
    df.reset_index(drop=True, inplace=True)
    print(len(df))

    # with open(os.path.join(temp_path, 'input-cand-user.pickle'), 'wb') as tf:
    #     pickle.dump(df, tf)
    #     tf.close()
    #
    # with open(os.path.join(temp_path, 'input-cand-user.pickle'), 'rb') as tf:
    #     x = pickle.load(tf)
    #     print(x)
    #     tf.close()

    # with open(os.path.join(temp_path, 'input-cand.pickle'), 'rb') as tf:
    #     x = pickle.load(tf)
    #     print(x)
    #     tf.close()

def merge_keywords():
    """
    Merge and deduplicate tweets collected using multiple keywords assoiated with one event
    :return: [event].jsonl
    """
    # data = glob(os.path.join('..', '..', 'data_augmentation/christchurch-shooting/*.json'))
    data = glob(os.path.join('..', '..', 'data_augmentation/manchesterbombings/*.json'))
    count = 0
    objs = []
    ids = set()
    duplicates = 0
    for f in data:
        with jsonlines.open(f) as reader:  # load tweets obtained uinsg Hydrator (.jsonl)
            for i, obj in enumerate(reader):
                # print(obj.keys())
                objs.append(obj)
                count += 1
                if obj['id'] in ids:
                    duplicates += 1 # Count duplicates
                    continue
                else: # Save unique tweets
                    # with jsonlines.open(os.path.join('..', '..', 'data_augmentation/christchurch-shooting/christchurch.jsonl'), mode='a') as writer:
                    with jsonlines.open(os.path.join('..', '..', 'data_augmentation/manchesterbombings/manchesterbombings.jsonl'), mode='a') as writer:
                        writer.write(obj)
                ids.add(obj['id'])
        reader.close()
    print("Total number of tweets before deduplication ", count)
    print("Number of duplicates ", duplicates)

    count = 0
    # with jsonlines.open(os.path.join('..', '..', 'data_augmentation/christchurch-shooting/christchurch.jsonl')) as reader:  # load tweets obtained uinsg Hydrator (.jsonl)
    with jsonlines.open(os.path.join('..', '..', 'data_augmentation/manchesterbombings/manchesterbombings.jsonl')) as reader:  # load tweets obtained uinsg Hydrator (.jsonl)
        for i, obj in enumerate(reader):
            count +=1
        reader.close()
    print("Number of tweets after deduplication ", count)


def main():
    if platform == 'linux':
        parser =argparse.ArgumentParser
        parser.add_argument(
            '--event', default='germanwings',
            help='The name of event: default=%default')
        parser.add_argument(
            '--idname',  default='2015-germanwings-crash.ids',
            help='The name of the file containing Tweet ids: default=%default')
        parser.add_argument(
            '--tweets', default='germanwings.jsonl',
            help='The name of the file containing hydrated Tweets (.jsonl): default=%default')
        args = parser.parse_args()
        event = args.event
        idname = args.idname
        tweets = args.tweets

    elif platform == 'darwin':
        event ='germanwings'
        print(event)
        tweets = '{}.jsonl'.format(event)

    convert_downloaded_data(event,  tweets)
    # merge_keywords()

main()


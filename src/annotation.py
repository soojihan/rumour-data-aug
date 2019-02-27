import os
import json
import pandas as pd
import numpy as np
from pprint import pprint as pp
from textcleaner import tokenize_by_word
import re
import spacy
import sys

nlp = spacy.load('en_core_web_lg')
pd.set_option('display.max_columns', None)
doc1 = nlp('break bomb report near finish line add story develop')
# doc2 = nlp('break authority investigate repo two explosion finish line')
doc2 = nlp('break report several people injured explosion finish line')
print(doc1.similarity(doc2))

def load_csv(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)

    return df

def segment_csv():
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_reference.csv')
    ref = load_csv(ref_path)
    ref.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
    ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    # ref = ref[2:]
    ref.reset_index(inplace=True, drop=True)
    print(len(ref))
    print(ref.head(2))
    print(ref.tail(2))
    num_segmentations = list(np.arange(17)*12)
    print(num_segmentations)

    for i, num in enumerate(num_segmentations):
        if i>0:
            print(num_segmentations[i-1] ,num)
            subset = ref.loc[num_segmentations[i-1]: min(num, len(ref))-1]
            print(subset)
            subset.reset_index(inplace=True, drop=True)
            out = os.path.join('..', 'train_augmentation/data')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'boston_{}.csv'.format(i)), 'w') as f:
                subset.to_csv(f)
        else:
            continue

def semantic_similarity():
    # ref_path = os.path.join('..', 'bostonbombings_reference.csv')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_reference.csv')
    # ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_augmentation/data/boston_{}.csv'.format(batch_num))

    # data_path = os.path.join('..','bostonbombings_candidates.csv')
    # data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_candidates_with_rts.csv')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_candidates.csv')

    ref = load_csv(ref_path)
    data = load_csv(data_path)
    ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    data.drop(['Unnamed: 0', 'Unnamed: 0.1'], inplace=True, axis=1)
    ref.dropna(inplace=True)
    data.dropna(inplace=True)
    print("the number of rows in the original data: ", len(data))
    print("the number of rows in the original reference: ", len(ref))

    data.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
    ref.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
    print(list(ref))
    print(list(data))

    print("the number of rows in the deduplicated data: ", len(data))
    print("the number of rows in the deduplicated reference: ", len(ref))

    result_columns = ['ref_original', 'ref_processed', 'sim_score', 'status', 'id',
                      'screen_name', 'text', 'created_at', 'retweet_count', 'processed_text']
    result_df = pd.DataFrame(columns=result_columns)
    for i, ref_row in ref.iterrows():
        # ref_tweet = nlp(ref_row['original_text'])
        ref_tweet = ref_row['processed_text']
        print(ref_tweet)
        ref_tweet = nlp(ref_tweet)
        for j, data_row in data.iterrows():
            print(i, j, "/" ,len(data))
            # print(list(data_row))
            tweet = data_row['processed_text']
            print(tweet)
            tweet = nlp(tweet)


            sim_score = ref_tweet.similarity(tweet)
            if sim_score > 0.5:
                print("** ref: ", ref_tweet)
                print("** tweet: ", tweet)
                print(sim_score)

                result_df.loc[len(result_df)] = [ref_row['original_text'], ref_row['processed_text'], sim_score]+list(data_row)
            # print(result_df)
            print("")
            if j==10:
                break
        if i==3:
            break
        # result_df.to_csv('bostonbombings_results.csv')
        # result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'train_augmentation/sim_scores')
        # os.makedirs(result_path, exist_ok=True)
        # result_df.to_csv(os.path.join(result_path, 'boton_{}.csv'.format(batch_num)))
        # print("... Complete")

semantic_similarity()
# segment_csv()
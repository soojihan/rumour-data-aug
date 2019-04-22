import pandas as pd
import os
from glob import glob
from src.semantic_relatedness import load_data
import random
import numpy as np
import pickle
from collections import Counter
from pprint import pprint as pp
import jsonlines

def data_augmentation(event: str = 'sydneysiege'):
    """
    Deduplicate the final output (after filtering out by applying a threshold)
    Balance pos and eng examples using the threshold fine tuned usnig the SemEval
                                     threshold   precision
    # subset = df[df['sim_score'] >= 0.652584] # 6088
    # subset = df[df['sim_score'] >= 0.691062] # 7000
    # subset = df[df['sim_score'] >= 0.708341] # 7500
    # subset = df[df['sim_score'] >= 0.760198] # 8502
    # subset = df[df['sim_score'] >= 0.801806] # 9003
    # subset = df[df['sim_score'] >= 0.849272] # 9506

    :return:
    """

    ## Load tweet objects to collect ids of tweets with retweets
    # jsonl_file = os.path.join('..', 'data_augmentation/data_hydrator/downloaded_data/hydrator/{}.jsonl'.format(event))
    jsonl_file = os.path.join('..', 'data_augmentation/data_hydrator/downloaded_data/hydrator/charliehebdo_chunk/{}_11.jsonl'.format(event))
    jsonl_reader = jsonlines.open(jsonl_file)
    has_retweets = set()
    with jsonlines.open(jsonl_file) as reader:  # load tweets obtained uinsg Hydrator (.jsonl)
        for i, obj in enumerate(reader):
            if obj['retweet_count'] >0:
                has_retweets.add(obj['id_str'])

        reader.close()
    print("Number of tweets that have retweets ", len(has_retweets))


    # raise SystemExit
    # files = glob(os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/{}/scores/ref*.csv'.format(event)))
    files = glob(os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data_hydrator/file-embed-output/{}/scores_11/ref*.csv'.format(event)))
    assert len(files) == 61
    pos_ids = set()
    neg_ids = set()

    pos_threshold = 0.801806
    precision = 9003
    neg_threshold = 0.266
    pos_scores={}
    neg_scores={}
    # files = files[:2]
    print(len(files))
    for findex, f in enumerate(files): # Iter results for each reference tweet
        df = pd.read_csv(f)
        print(findex)

        ## Select positive examples using a fine-tuned threshold
        pos_subset = df[df['sim_score'] >= pos_threshold]
        ## Sample positive examples. 'id' here refers to the indices of a candidate set
        sub_pos_ids = set(pos_subset['id'].astype(int).values)
        ## Sample negative examples
        # TODO: Improve a method for sampling negative examples
        neg_subset = df[df['sim_score'] < neg_threshold]
        sub_neg_ids = set(neg_subset['id'].astype(int).values)

        # Update sampled ids.
        pos_ids.update(sub_pos_ids)
        neg_ids.update(sub_neg_ids)
        print("Updated negative indices ", len(neg_ids))

        ## Generate a dictionary semantic relatedness scores
        for pi, prow in pos_subset.iterrows():
            pid = int(prow['id'])
            if pid not in pos_scores:
                pos_scores[pid] = prow['sim_score']
            elif (pid in pos_scores) and (pos_scores[pid]<prow['sim_score']):
                pos_scores[pid] = prow['sim_score']
            else:
                continue

        for ni, nrow in neg_subset.iterrows():
            nid = int(nrow['id'])
            if nid not in neg_scores:
                neg_scores[nid] = nrow['sim_score']
            elif (nid in neg_scores) and (neg_scores[nid]<nrow['sim_score']):
                neg_scores[nid] = nrow['sim_score']
            else:
                continue
        print("")

    print("*"*10)
    print("Number of positive examples ", len(pos_ids))
    print("Number of positive scores ", len(pos_scores.items()))
    print("Number of negative examples ", len(neg_ids))
    print("Number of negative scores ", len(neg_scores.items()))
    total_scores = pos_scores.copy()
    total_scores.update(neg_scores)
    print(len(total_scores))

    print("")

    # infile = '/Users/suzie/Desktop/PhD/Projects/data-aug/data_augmentation/data/file-embed-output/{}/'.format(event)
    infile = os.path.join('..', 'data_augmentation/data_hydrator/file-embed-output/{}/'.format(event))
    _, cand = load_data(event=event, batch_num=11)
    pp(list(cand))
    # cand.drop(['index'], axis=1, inplace=True) ## TODO: add drop=True when generating candidate set in elmo_data_preprocessing.py
    # cand['order_index'] = cand['index'].values ## for separated files, keept the indices in the original candidate set
    cand['index'] = np.arange(len(cand)) ## score dfs' ids are associated with row numbers of the candidate set
    assert len(cand)==len(df)
    print(cand.index)

    ### full_df = pd.read_csv(os.path.join(infile, 'dropped_candidates.csv'))
    ### total_indices = list(full_df.index)
    ### print("Total number of candidates ", len(total_indices))


    ## Generate final input df
    # cand.drop(['Unnamed: 0'], inplace=True, axis=1)

    ## Leave tweet ids which have high simliarity scores and retweets
    pos_subset = cand[(cand.index.isin(pos_ids))&(cand.id.isin(has_retweets))]
    # pos_subset = cand[(cand.index.isin(pos_ids))]
    pos_subset['label'] = np.ones(len(pos_subset), dtype=int)
    print("--------------------------------------")
    print("Number of positive sampes ", len(pos_subset))
    ## Check duplicates
    ids = pos_subset['id']
    duple = pos_subset[ids.isin(ids[ids.duplicated()])].sort_values("id")
    print("Positive duplicates ", len(duple))

    num_pos = len(pos_subset)

    ## Set the number of negative samples to be saved
    num_sample = min(num_pos * 3, len(neg_ids))
    random_neg_indices = random.sample(neg_ids, num_sample)

    ## Find intersection -> if a tweet appears in both negative and positive samples, remove it from negative set
    intersection = pos_ids.intersection(random_neg_indices)
    print("before removing intersection ", len(intersection), len(random_neg_indices))
    random_neg_indices = set(random_neg_indices) - intersection
    print("after removing intersection ", len(random_neg_indices))
    # assert len(total_scores.items()) == (len(pos_scores.items())+ len(neg_scores.items()))

    # neg_subset = cand.loc[random_neg_indices]
    neg_subset = cand[cand['index'].isin(random_neg_indices)]
    # neg_subset = neg_subset[neg_subset['retweet_count']<100] # filter out negative subset
    neg_subset['label'] = np.zeros(len(neg_subset), dtype=int)

    ## Check duplicates
    ids = neg_subset['id']
    duple = neg_subset[ids.isin(ids[ids.duplicated()])].sort_values("id")
    print("Negative duplicates ", len(duple))

    result = pd.concat([pos_subset, neg_subset], sort=True)
    print(len(result))
    print(result.head())
    print("")
    for k, row in result.iterrows():
        if k in total_scores:
            result.loc[k, 'score'] = total_scores[k]
        else:
            raise ValueError
    result['id'] = result['id'].astype(str)

    # save_path = os.path.join(infile, 'results_p{}<0.3'.format(precision))
    # os.makedirs(save_path, exist_ok=True)
    result.to_csv(os.path.join(infile, '{}-{}_11.csv'.format(precision, neg_threshold)))
    with open(os.path.join(infile, '{}-{}_11.pickle'.format(precision, neg_threshold)), 'wb') as f:
        pickle.dump(result, f)

def manual_inspection(event: str):
    infile = os.path.join('..', 'data_augmentation/data_hydrator/file-embed-output/{}/'.format(event))
    with open(os.path.join(infile, '{}-{}.pickle'.format(9003, 0.266)), 'rb') as f:
        df = pickle.load(f)
    ids = df['id']
    duple = df[ids.isin(ids[ids.duplicated()])].sort_values("id")
    print(len(duple))

    pos = df[df.label == 1]
    random_ids = random.sample(list(pos['index'].values), 30)
    pos = pos[pos.index.isin(random_ids)]
    for t, s in zip(pos.text.values, pos.score.values):
        print(t, s)
        print("")
    print("*" * 20)
    pos = df[df.label == 0]
    random_ids = random.sample(list(pos['index'].values), 30)
    pos = pos[pos.index.isin(random_ids)]
    for t, s in zip(pos.text.values, pos.score.values):
        print(t, s)
        print("")

event = 'charliehebdo'

data_augmentation(event=event)
# manual_inspection(event)

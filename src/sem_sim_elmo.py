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
from textcleaner import tokenize_by_word

## ElmoEmbedder version
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Use the 'Small' pre-trained model
# options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/' \
#                '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
# weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo' \
#               '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

elmo = ElmoEmbedder(options_file= options_file, weight_file=weight_file)


# batch_num = sys.argv[1]
batch_num = 1

def load_csv(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)

    return df

def filter_candidate_tweets():
    """
    Filter candidates based on the number of tokens
    :return:
    """
    # ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_reference.csv')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_augmentation/data/boston_{}.csv'.format(batch_num))

    # data_path = os.path.join('..','bostonbombings_candidates.csv')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'boston_candidates_rts_more-than-4-tokens.csv')

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

    new_df = pd.DataFrame(columns=list(data))
    for i, row in data.iterrows():
        if i%1000==0:
            print(i)
        if len(row.processed_text.split())>4:
            new_df.loc[len(new_df)] = row
        else:
            continue
    new_df.to_csv('boston_candidates_rts_more-than-4-tokens.csv')
    data = new_df
    data.reset_index(inplace=True, drop=True)
    ref.reset_index(inplace=True, drop=True)
    print(list(ref))
    print(list(data))


    print("the number of rows in the deduplicated data: ", len(data))
    print("the number of rows in the deduplicated reference: ", len(ref))
    result_columns = ['ref_original', 'ref_processed', 'sim_score', 'status', 'id',
                      'screen_name', 'text', 'created_at', 'retweet_count', 'processed_text']
    result_df = pd.DataFrame(columns=result_columns)
    candidates = dict(zip(data.index, data.processed_text))

    for key, value in candidates.items():
        print(key, value)
        raise SystemExit
    # result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'train_augmentation/sim_scores')
    # os.makedirs(result_path, exist_ok=True)

def elmo_semantic_similarity():
    # ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bostonbombings_reference.csv')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_augmentation/data/boston_{}.csv'.format(batch_num))

    # data_path = os.path.join('..','bostonbombings_candidates.csv')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'boston_candidates_rts_more-than-4-tokens.csv')

    ref = load_csv(ref_path)
    data = load_csv(data_path)
    # data=data[:100]
    # ref=ref[:12]
    print(len(data))
    print(list(ref))
    print(list(data))
    ref.drop(['Unnamed: 0'], inplace=True, axis=1)
    data.drop(['Unnamed: 0'], inplace=True, axis=1)
    ref.dropna(inplace=True)
    data.dropna(inplace=True)
    print("the number of rows in the original data: ", len(data))
    print("the number of rows in the original reference: ", len(ref))

    tokenised_candidates = list(map(lambda x: word_tokenize(x), data['processed_text'].values))
    tokenised_ref = list(map(lambda x: word_tokenize(x), ref['processed_text'].values))
    # pp(tokenised_ref)

    start = time.time()
    ## Computes the ELMo embeddings for a batch of tokenized sentences.
    elmo_candidates = elmo.embed_batch(tokenised_candidates)
    elmo_ref = elmo.embed_batch(tokenised_ref)

    ## Compute the mean elmo vector for each tweet
    elmo_candidates_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_candidates))
    elmo_ref_avg = list(map(lambda x: np.mean(x[2],axis=0).reshape(1,-1), elmo_ref))
    elmo_candidates_avg = np.squeeze(np.asarray(elmo_candidates_avg), axis=1)
    print(elmo_candidates_avg.shape)
    elmo_ref_avg = np.squeeze(np.asarray(elmo_ref_avg), axis=1)
    print(elmo_ref_avg.shape)

    # print("cosine", cosine_similarity(elmo_ref_avg, elmo_candidates_avg))
    print("cosine sim shape", cosine_similarity(elmo_ref_avg, elmo_candidates_avg).shape)
    target_indices =  np.where(cosine_similarity(elmo_ref_avg, elmo_candidates_avg)>0.5)
    print(target_indices)

    for x in target_indices:
        print("x")
        print(x)

    ids = [(ref_id, cand_id) for ref_id, cand_id in zip(target_indices[0], target_indices[1])]
    pp(ids)
    elmo_indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_augmentation', 'sim_scores'))
    print(elmo_indices_path)
    with open(os.path.join(elmo_indices_path, 'sem_sim_indices_{}.pickle'.format(batch_num)), 'wb') as f:
        pickle.dump(ids, f)
        f.close()

    with open(os.path.join(elmo_indices_path, 'sem_sim_scores_{}.pickle'.format(batch_num)), 'wb') as f:
        pickle.dump(cosine_similarity(elmo_ref_avg, elmo_candidates_avg), f)
        f.close()

    end=time.time()
    print("Time elapsed ", end-start)


# elmo_semantic_similarity()
raise SystemExit


print("")
elmo_indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_augmentation', 'sim_scores'))
print(elmo_indices_path)

with open(os.path.join(elmo_indices_path, 'sem_sim_indices_{}.pickle'.format(batch_num)), 'rb') as f:
    indices = pickle.load(f)
    f.close()

with open(os.path.join(elmo_indices_path, 'sem_sim_scores_{}.pickle'.format(batch_num)), 'rb') as f:
    scores = pickle.load(f)
    f.close()


print("loaded semantic similarity scores shape: ", scores.shape)
# print(scores)
target_indices = np.where((scores > 0.5)&(scores<0.7))
indices = [(ref_id, cand_id,scores[ref_id, cand_id] ) for ref_id, cand_id in zip(target_indices[0], target_indices[1])]
indices = sorted(indices, key=operator.itemgetter(2), reverse=False)

ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_augmentation/data/boston_{}.csv'.format(batch_num))

# data_path = os.path.join('..','bostonbombings_candidates.csv')
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'boston_candidates_rts_more-than-4-tokens.csv')

ref = load_csv(ref_path)
data = load_csv(data_path)
data=data[:1000]
ref=ref[:12]
print(len(data))
print(list(ref))
print(list(data))
ref.drop(['Unnamed: 0'], inplace=True, axis=1)
data.drop(['Unnamed: 0'], inplace=True, axis=1)
ref.dropna(inplace=True)
data.dropna(inplace=True)
print("the number of rows in the original data: ", len(data))
print("the number of rows in the original reference: ", len(ref))

data.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
ref.drop_duplicates(subset = ['processed_text'], inplace=True) # remove duplicate rows
print(len(data))
print(len(ref))
# for ids in indices:
#     ref_id = ids[0]
#     cand_id = ids[1]
#     sim_score = ids[2]
#     print(ref_id, cand_id, sim_score, ref.loc[ref_id, 'original_text'])
#     print(data.loc[cand_id, 'text'])
#     print("")


## ElmoEmbedder version
# elmo = ElmoEmbedder(options_file= options_file, weight_file=weight_file)
# tokens1 = ["I", "ate", "an", "apple", "for", "breakfast"]
# tokens2 = ["She", "had", "many", "strawberries", "dinner"]
#
# # tokens2 = ["She", "had", "many", "strawberries", "dinner"]
# start = time.time()
# embeddings1 = elmo.embed_sentence(tokens1)
# embeddings2 = elmo.embed_sentence(tokens2)
# print(embeddings1.shape)
# v1 = np.squeeze(embeddings1)
# v2 = np.squeeze(embeddings2)
# print(cosine_similarity(v1[2], v2[2]))
# print(sp.spatial.distance.cosine(v1[2,0,:],v2[2,0,:]))
#
# v1_avg = np.mean(v1[2], axis=0)
# print(v1_avg.shape)
# v2_avg = np.mean(v2[2], axis=0)
# print(v2_avg.shape)
# print("cosine", cosine_similarity(v1_avg.reshape(1,-1), v2_avg.reshape(1,-1)))
#
# end=time.time()
# print(end-start)



# ## Tensorflow Hub version
#
# tokens2 = ["She", "had", "many", "strawberries", "dinner", ""]
#
# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# tokens_input = [tokens1, tokens2]
# tokens_length = [6, 5]
# start = time.time()
# embeddings = elmo(
# inputs={
# "tokens": tokens_input,
# "sequence_len": tokens_length
# },
# signature="tokens",
# as_dict=True)["elmo"]
#
# # Convert tensor to numpy array
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     elmo_embeddings = sess.run(embeddings)
#     print(type(elmo_embeddings))
# v1 = elmo_embeddings[0,:,:]
# print(v1.shape)
# print(type(v1))
# v2 = elmo_embeddings[1,:,:]
# print(v2.shape)
# v1_avg = np.mean(v1, axis=0)
# print(v1_avg.shape)
# v2_avg = np.mean(v2, axis=0)
# print(v2_avg.shape)
# print("cosine", cosine_similarity(v1_avg.reshape(1,-1), v2_avg.reshape(1,-1)))
# end=time.time()
# print(end-start)


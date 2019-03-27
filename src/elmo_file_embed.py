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
# from preprocessing import text_preprocessor

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
from sys import platform
from typing import IO, List, Iterable, Tuple

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


def load_elmo(finetuned: bool =True):
    """
    Load ElMo
    :return:
    """
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    if finetuned:
        ## Load weights fine-tuned using CREDBANK
        # weight_file = os.path.join(os.path.dirname(__file__), '..', '..', 'rumourdnn', "resource", "embedding",
        #                                     "elmo_model", "weights_12262018.hdf5")
        weight_file = '/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/resource/embedding/weights_12262018.hdf5'
    else:
        ## Load pre-trained weights
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    global elmo
    elmo = ElmoEmbedder(
        options_file= options_file,
        weight_file= weight_file)


def get_elmo_embeddings(infile: IO,
                        outfile: str,
                        output_format: str = 'average'):
    """
    Compute Elmo embeddings
    :return: hdf5 file (ELMo embeddings per sentence)
    """
    elmo.embed_file(input_file=infile, output_file_path=outfile, output_format=output_format)


def main():
    if platform == 'darwin':
        event = 'sydneysiege'
        infile_cand = os.path.join('..',
                              'data_augmentation/data/file-embed-input/{}/input-cand-noempty.txt'.format(event))
        outfile_cand = os.path.join('..',
                               'data_augmentation/data/file-embed-output/{}/output-cand.hdf5'.format(event))
        outfile_cand = os.path.abspath(outfile_cand)

        infile_ref = os.path.join('..',
                                   'data_augmentation/data/file-embed-input/{}/input-ref.txt'.format(event))
        outfile_ref = os.path.join('..',
                                    'data_augmentation/data/file-embed-output/{}/output-ref.hdf5'.format(event))
        outfile_ref = os.path.abspath(outfile_ref)
        elmo_format = 'average'

    elif platform == 'linux':
        parser = argparse.ArgumentParser()
        parser.add_argument('--event', help='the name of event')
        parser.add_argument('--infile_cand', help='path to candidates')
        parser.add_argument('--infile_ref', help='path to ref')
        parser.add_argument('--outfile_cand', help='path to store cand embedding')
        parser.add_argument('--outfile_ref', help='path to store ref embedding')
        parser.add_argument('--elmo_format', help='The embeddings to output.  Must be one of "all", "top", or "average"')
        # print(parser.format_help())

        args = parser.parse_args()
        event = args.event
        infile_cand = args.infile_cand
        infile_ref = args.infile_ref
        outfile_cand = args.outfile_cand
        outfile_ref = args.outfile_ref
        elmo_format = args.elmo_format

    load_elmo()
    get_elmo_embeddings(infile=infile_ref, outfile = outfile_ref, output_format=elmo_format)
    get_elmo_embeddings(infile=infile_cand, outfile = outfile_cand, output_format=elmo_format)


main()
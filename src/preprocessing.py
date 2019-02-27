import os
import json
import pandas as pd
import numpy as np
from pprint import pprint as pp
from textcleaner import tokenize_by_word
import re
import spacy


# nlp = spacy.load('en_core_web_sm')
# pd.set_option('display.max_columns', None)
#
#
# doc1 = nlp('break bomb report near finish line add story develop')
# # doc2 = nlp('break authority investigate repo two explosion finish line')
# doc2 = nlp('break report several people injured explosion finish line')
# print(doc1.similarity(doc2))

def text_preprocessor(text, language='english', deaccent=True, ignore_retweet = False):
    original_text = text
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    processed_text = text.lower()
    ### Ignore retweets
    final_text = None

    if ignore_retweet and len(re.findall(r'^(rt)( @\w*)?[: ]', processed_text)) != 0:  # is retweet?
        final_text = None

    else:
        processed_text = re.sub(r'^(rt)( @\w*)?[: ]', '',
                                processed_text)  # remove 'rt @username' from tweet; when considering retweets
        processed_text = re.sub(r"http\S+\s*", "", processed_text)  # remove URL
        processed_text = re.sub(r"pic.twitter.com\S+", "", processed_text)  # remove pic URL
        split_text = list(tokenize_by_word(processed_text))
        split_text = list(filter(None, split_text))  # filter empty strings

        # if len(split_text) < 4: # bostonbombings
        if len(split_text) < 1: # semeval
            final_text = None
        else:
            final_text = " ".join(split_text)

    return final_text



def load_raw_tweets(file_path):
    """
    load tweet event 2012-2016 data
    :param file_path: path to csv file
    :return: csv
    """
    with open(file_path, 'r') as f:
        original_data = pd.read_csv(f)
        f.close()
    return original_data

def load_crisislex_references(file_path):
    """
    load reference csv file
    :param file_path: path to csv file
    :return: csv
    """
    with open(file_path, 'r') as f:
        ref = pd.read_csv(f)
        f.close()
    ref = ref[ref[' Informativeness'] == 'Related and informative']
    return ref

def generate_clean_reference():
    """
    Generate a collection of reference tweets after removing irrelevant categories
    :return: dataframe containing original reference tweet and pre-processed ref. tweet
    """
    info_type = [
     'Affected individuals',
     # 'Caution and advice',
     # 'Donations and volunteering',
     'Infrastructure and utilities',
     'Not applicable',
     'Other Useful Information',
     # 'Sympathy and support'
    ]

    ref_infile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/reference/bostonbombings.csv')
    ref = load_crisislex_references(ref_infile)
    outdf = pd.DataFrame(columns=['text'])
    processed_corpus = []
    original_corpus = []
    for t in info_type:
        subset = ref[ref[' Information Type']==t]
        gt_corpus = list(map(lambda x: text_preprocessor(x, ignore_retweet=False), subset[' Tweet Text'].values))
        assert len(gt_corpus)==len(subset[' Tweet Text'].values)
        zipped_corpus = zip(subset[' Tweet Text'].values, gt_corpus)

        for o, p in zipped_corpus:
            if p is not None:
                processed_corpus.append(p)
                original_corpus.append(o)

    outdf['original_text'] = original_corpus
    outdf['processed_text'] = processed_corpus
    print(outdf)

    outfile = '../{}.csv'.format('bostonbombings')
    outdf.to_csv(outfile)

    ## Remove nan rows and reindex dataframe after manually removing irrelevant reference tweets
    # with open('../bostonbombings_reference.csv', 'r') as f:
    #     data = pd.read_csv(f)
    # data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # data.dropna(axis=0, inplace=True)
    # data.reset_index(inplace=True)
    # data.drop(['index'], axis=1, inplace=True)
    # print(data)
    # outfile = '../{}.csv'.format('bostonbombings_reference')
    # data.to_csv(outfile)

def generate_clean_candidates():
    """
    Remove retweets from the original data in order to find source tweets from Tweet Event 2012-2016 dataset
    :return: csv
    """
    path_to_dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/general/bostonbombings.csv')

    original_data = load_raw_tweets(path_to_dataset)
    # original_data = original_data[10000:10100]
    print(len(original_data))
    # pp(list(original_data))
    original_corpus = list(map(lambda x: text_preprocessor(x, ignore_retweet=False), original_data['text'].values))
    # original_corpus = [x for x in original_corpus if x is not None]
    # print(len(original_corpus))
    # pp(original_corpus)
    assert len(original_corpus) == len(original_data)
    original_data['processed_text'] = original_corpus
    # print(original_data)

    # outfile = '../{}.csv'.format('bostonbombings_candidates')
    outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), '{}.csv'.format('bostonbombings_candidates_with_rts'))
    original_data.to_csv(outfile)
    print("Done")

# generate_clean_candidates()
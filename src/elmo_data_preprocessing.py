from semeval.semeval_data_processor import load_csv
import os
import pandas as pd
from credbankprocessor import preprocessing_tweet_text
from pprint import pprint as pp
from typing import IO, List, Iterable, Tuple
import pickle
import numpy as np

def load_data(event: str = None,
              name: str = 'pheme'):
    print("Loading data...")
    if name=='pheme':
        ref = os.path.join('..',
                           'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
        cand = os.path.join('..',
                            'data_augmentation/data/candidates/{}.csv'.format(event))
        # ref = os.path.join(ref_path, '{}.csv'.format(event))
        # cand = os.path.join(data_path, '{}.csv'.format(event))
        ref = load_csv(ref)
        data = load_csv(cand)
        ref = ref[['text']]
        data.drop(['Unnamed: 0'], inplace=True, axis=1)
        ref.dropna(inplace=True)
        data.dropna(inplace=True)
        ref.reset_index(inplace=True, drop=True)
        data.reset_index(inplace=True, drop=True)
        print("the number of rows in the original data: ", len(data))
        print("the number of rows in the original reference: ", len(ref))

        return ref, data

    elif name=='semeval':
        data_type = "merged_semeval"
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'data/semeval2015/data/{}.csv'.format(data_type))
        data = load_csv(data_path)
        data.drop(['Unnamed: 0'], inplace=True, axis=1)
        if data.isnull().any().any():
            raise ValueError
        print("the number of rows in the original data: ", len(data))

        return data

    elif name=='augmented':
        data = os.path.join('..', 'data_augmentation/data_hydrator/saved_data/source-tweets/{}/input-cand.pickle'.format(event))
        ref = os.path.join('..', 'data_augmentation/data/pheme_annotations/{}-all-rnr-threads.csv'.format(event))
        print(data)
        print(ref)
        with open(os.path.join(data), 'rb') as tf:
            cand = pickle.load(tf)
            tf.close()
        ref = load_csv(ref)
        ref = ref[['text', 'label']]
        ref = ref[ref['label']=='1']
        ref.dropna(inplace=True)
        ref.reset_index(inplace=True, drop=True)
        print("Number of rows in the original reference: ", len(ref))
        return cand, ref

def preprocess_tweets(tweets):
    print("Number of tweets before preprocessing ", len(tweets.values))
    processed_text = list(map(lambda x: preprocessing_tweet_text(x), tweets.values))
    tokenised_text = list(map(lambda x: " ".join(x), processed_text))
    tokenised_text = list(map(lambda x: x.replace("\r", " ").replace("\n", " ").strip("/"), tokenised_text))
    blank_lines = [i for (i, line) in enumerate(tokenised_text) if line == ""]
    print("Number of tweets after preprocessing ", len(tokenised_text))
    print("Number of empty strings: ", len(blank_lines))
    assert len(tweets) == len(tokenised_text)

    return tokenised_text, blank_lines

def preprocess_main(name: str,
                  event: str):
    """
    Generate input files for ELMO embed_file method; each line contains a sentence tokenized by whitespace.
    :return: text files (input to ELMo)
    """
    if name == 'semeval':
        data = load_data(name=name)
        outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input')
        tokenised_tweet_cand = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet1'].values))
        tokenised_tweet_ref = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet2'].values))
        outfile = os.path.abspath(outfile)
        os.makedirs(outfile, exist_ok=True)
        print(outfile)

        for t in tokenised_tweet_cand:
            with open(os.path.join(outfile, 'input-cand.txt'), 'a') as f:
                f.write(t)
                f.write("\n")
        f.close()

        for t in tokenised_tweet_ref:
            with open(os.path.join(outfile, 'input-ref.txt'), 'a') as f:
                f.write(t)
                f.write("\n")
        f.close()

    elif name=='pheme': #TODO: Merge with 'augmented' when completing downloading tweets using Hydrator;
                        #TODO: Remove methods 'remove empty_indices' and 'remove_empty_strings'
        ref, data = load_data(name=name, event=event)
        outfile = os.path.abspath(
            os.path.join('..', 'data_augmentation/data/file-embed-input/{}'.format(event)))
        outfile = os.path.abspath(outfile)
        os.makedirs(outfile, exist_ok=True)
        print(outfile)

        processed_cand = list(map(lambda x: preprocessing_tweet_text(x), data['text'].values))
        processed_ref = list(map(lambda x: preprocessing_tweet_text(x), ref['text'].values))
        tokenised_tweet_cand = list(map(lambda x: " ".join(x), processed_cand))
        tokenised_tweet_ref = list(map(lambda x: " ".join(x), processed_ref))
        blank_lines_cand = [i for (i, line) in enumerate(tokenised_text_cand) if line == ""]
        blank_lines_ref = [i for (i, line) in enumerate(tokenised_text_ref) if line == ""]
        print("Number of candidate tweets after preprocessing ", len(tokenised_text_cand))
        print("Number of empty strings in candidate set: ", len(blank_lines_cand))
        print("Number of reference tweets after preprocessing ", len(tokenised_text_ref))
        print("Number of empty strings in reference set: ", len(blank_lines_ref))
        assert len(data) == tokenised_tweet_cand
        assert len(ref) == tokenised_tweet_ref
        data['processed_text'] = tokenised_tweet_cand
        ref['processed_text'] = tokenised_tweet_ref
        new_data = data.drop(blank_lines_cand)
        new_ref = ref.drop(blank_lines_ref)

        assert len(new_data) + len(blank_lines_cand) == len(data)
        assert len(new_ref) + len(blank_lines_ref) == len(ref)
        new_data.reset_index(inplace=True)
        new_ref.reset_index(inplace=True)
        with open(os.path.join(outfile, 'input-cand-processed.pickle'), 'wb') as tf:
            pickle.dump(new_data, tf)
            tf.close()
        with open(os.path.join(outfile, 'input-ref-processed.pickle'), 'wb') as tf:
            pickle.dump(new_ref, tf)
            tf.close()


    elif name=='augmented':
        data, ref = load_data(name=name, event=event)
        outfile = os.path.abspath(
            os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
        os.makedirs(outfile, exist_ok=True)
        print(outfile)

        tokenised_cand, cand_blank = preprocess_tweets(data['text'])
        tokenised_ref, ref_blank = preprocess_tweets(ref['text'])
        data['processed_text'] = tokenised_cand
        ref['processed_text'] = tokenised_ref
        new_cand = data.drop(cand_blank)
        new_ref = ref.drop(ref_blank)
        assert len(new_cand)+len(cand_blank) == len(data)
        assert len(new_ref)+len(ref_blank) == len(ref)
        print(len(new_ref))
        new_ref.drop_duplicates(keep='first', inplace=True)
        print(len(new_ref))
        new_cand.reset_index(inplace=True)
        new_ref.reset_index(inplace=True)
        with open(os.path.join(outfile, 'input-cand-processed.pickle'), 'wb') as tf:
            pickle.dump(new_cand, tf)
            tf.close()

        with open(os.path.join(outfile, 'input-ref-processed.pickle'), 'wb') as tf:
            pickle.dump(new_ref, tf)
            tf.close()
    else:
        print("Check data name")

    print("Done")

def prepare_input(outpath: str,
                  event: str):
    targets =['cand', 'ref']
    # targets =[ 'ref']
    for t in targets:
        with open(os.path.join(outpath, 'input-{}-processed.pickle'.format(t)), 'rb') as tf:
            df = pickle.load(tf)
            tf.close()
        print(len(df))
        print(df.head())

        with open(os.path.join(outpath, 'elmo_{}_input.txt'.format(t)), 'w') as f:
            for i, row in df.iterrows():
                f.write(str(row['processed_text']))
                f.write('\n')
            f.close()

        with open(os.path.join(outpath, 'elmo_{}_input.txt'.format(t)), 'r') as f:
            lines = f.read().splitlines()
            print(len(lines))
            assert len(lines) == len(df)
            # for i, line in enumerate(f):
            # for i, line in enumerate(lines):
            #     x = df.loc[i, 'processed_text']
            #     if x != line:
            #         raise SystemExit


def empty_indices(event: str,
                 outpath: str,
                  infile: str,
                  t: str ='candidates',
                  action: str ='save',
                  ): #TODO: REMOVE
    """
    ELMo embed_file raises error if there are empty strings --> remove
    :param: event
    :param: t: 'candidates' or 'ref'
    :param: action: whether to save or load empty strings' indices
    :return:
    """
    if action == 'save':
        # input_file = open(infile, 'r')

        # print(len(input_file.read().splitlines()))
        sentences = [line.strip() for line in input_file.readlines()]
        blank_lines = [i for (i, line) in enumerate(sentences) if line == ""]
        print("Total number of {} tweets: {} ".format(t, len(sentences)))
        print("Number of empty strings: ", len(blank_lines))

        print("")
        print("Saving the indices of empty strings...")
        with open(os.path.join(outpath, 'empty_index.pickle'.format(event, t)), 'wb') as f:
            pickle.dump(blank_lines, f)

    elif action =='load':
        print("Loading the indices of empty strings....")
        input_file = os.path.join(infile, 'empty_index.pickle')
        if os.path.exists(infile):
            with open(infile, 'rb') as f:
                ids = pickle.load(f)
            return ids
        else:
            return []


def remove_empty_strings(indices: IO,
                         event: str): #TODO: REMOVE
    """
    Remove empty sentences; ELMo file embed raises error when there're empty strings
    :param indices: indices of empty lines
    :return:
    """
    fpath = os.path.abspath(os.path.join('..',  'data_augmentation/data/file-embed-input/{}'.format(event)))

    input_file = open(os.path.join(fpath, 'input-cand.txt'), 'r')
    sentences = [line.strip() for line in input_file.readlines()]
    print(len(sentences))
    print(len(indices))
    original_len = len(sentences)
    # Save valid sentences
    sentences = [i for j, i in enumerate(sentences) if j not in indices]
    print(len(sentences))
    print(len(indices))
    assert original_len == (len(sentences)+len(indices))

    for i, t in enumerate(sentences):
        print(i, t)
        with open(os.path.join(fpath, 'input-cand-noempty.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
        f.close()


def main():
    event = 'germanwings'
    # preprocess_main(name='augmented', event=event)
    outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
    prepare_input(outpath=outpath, event=event)

    # outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
    ## outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data/{}'.format(t)))
    # infile = os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}/input-cand.txt'.format(event))
    # empty_indices(event=event, outpath=outpath, infile=infile, t='ref', action='save')
    #
    # remove_empty_strings(empty_indices(event='sydney', action='load', t='candidates'), event='sydneysiege')

if __name__ == '__main__':
    main()


from semeval.semeval_data_processor import load_csv
import os
import pandas as pd
from credbankprocessor import preprocessing_tweet_text
from pprint import pprint as pp
from typing import IO, List, Iterable, Tuple
import pickle
import numpy as np

def load_data(event: str = None,
              name: str = 'pheme',
              cand: bool = True,
              ref: bool = True):
    print("Loading data...")
    if name=='pheme':
        if ref:
            ref_d = os.path.join('..',
                           'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
            # ref = os.path.join(ref_path, '{}.csv'.format(event))
            ref_d = load_csv(ref_d)
            ref_d = ref_d[['text']]
            ref_d.dropna(inplace=True)
            ref_d.reset_index(inplace=True, drop=True)
            print("Number of original references: ", len(ref_d))

        if cand:
            cand_d = os.path.join('..',
                            'data_augmentation/data/candidates/{}.csv'.format(event))
            # cand = os.path.join(data_path, '{}.csv'.format(event))
            data = load_csv(cand_d)
            data.drop(['Unnamed: 0'], inplace=True, axis=1)
            data.dropna(inplace=True)
            data.reset_index(inplace=True, drop=True)
            print("Number of original candidates: ", len(data))

        if ref and cand:
            return ref_d, data
        elif ref and not cand:
            return ref_d
        elif cand and not ref:
            return data
        else:
            print("Check dataset type ")


    elif name=='semeval':
        data_type = "merged_semeval"
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'data/semeval2015/data/{}.csv'.format(data_type))
        data = load_csv(data_path)
        data.drop(['Unnamed: 0'], inplace=True, axis=1)
        if data.isnull().any().any():
            raise ValueError
        print("Number of original paraphrase pairs: ", len(ref))

        return data

    elif name=='augmented':
        if ref:
            # ref = os.path.join('..', 'data_augmentation/data/pheme_annotations/{}-all-rnr-threads.csv'.format(event))
            # ref = os.path.join('..', 'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
            ref_d = os.path.join('..', 'data_augmentation/data/ref/{}.csv'.format(event))
            print(os.path.abspath(ref_d))
            ref_d = load_csv(ref_d)
            ref_d = ref_d[['text']]
            # ref = ref[['text', 'label']]
            # ref = ref[ref['label']=='1']
            ref_d.dropna(inplace=True)
            ref_d.reset_index(inplace=True, drop=True)
            print("Number of original references: ", len(ref_d))

        if cand:
            data = os.path.join('..', 'data_augmentation/data_hydrator/saved_data/source-tweets/{}/input-cand.pickle'.format(event))
            with open(os.path.join(data), 'rb') as tf:
                data = pickle.load(tf)
                tf.close()
            print("Number of original candidates: ", len(data))

        if ref and cand:
            return ref_d, data
        elif ref and not cand:
            return ref_d
        elif cand and not ref:
            return data
        else:
            print("Check dataset type ")

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
                    event: str,
                    ref: bool = True,
                    cand: bool = True):
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

    elif (name=='augmented') or (name=='pheme'):
        ref_d = load_data(name=name, event=event, cand = cand, ref= ref)
        # data = load_data(name=name, event=event, cand=True, ref=False)
        outfile = os.path.abspath(
            os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
            # os.path.join('..', 'data_augmentation/data/file-embed-input/{}'.format(event)))
        os.makedirs(outfile, exist_ok=True)
        print(outfile)

        if ref:
            tokenised_ref, ref_blank = preprocess_tweets(ref_d['text'])
            ref_d['processed_text'] = tokenised_ref
            new_ref = ref_d.drop(ref_blank)
            assert len(new_ref) + len(ref_blank) == len(ref_d)
            new_ref.reset_index(inplace=True)

            with open(os.path.join(outfile, 'input-ref-processed.pickle'), 'wb') as tf:
                pickle.dump(new_ref, tf)
                tf.close()

        if cand:
            tokenised_cand, cand_blank = preprocess_tweets(data['text'])
            data['processed_text'] = tokenised_cand
            new_cand = data.drop(cand_blank)
            assert len(new_cand)+len(cand_blank) == len(data)
            new_cand.reset_index(inplace=True)

            with open(os.path.join(outfile, 'input-cand-processed.pickle'), 'wb') as tf:
                pickle.dump(new_cand, tf)
                tf.close()
    else:
        print("Check data name")

    print("Done")

def prepare_input(outpath: str,
                  event: str,
                  cand: bool = True,
                  ref: bool = True):
    if ref and cand:
        targets =['cand', 'ref']
    elif ref and not cand:
        targets =[ 'ref']
    elif cand and not ref:
        targets =['cand']
    else:
        print("Check dataset type ")
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



def main():
    event = 'manchesterbombings'
    # preprocess_main(name='augmented', event=event, cand=False)
    outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
    prepare_input(outpath=outpath, event=event, cand=False)


if __name__ == '__main__':
    main()


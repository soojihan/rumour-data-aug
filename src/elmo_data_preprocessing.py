from semeval.semeval_data_processor import load_csv
import os
import pandas as pd
from credbankprocessor import preprocessing_tweet_text
from pprint import pprint as pp
from typing import IO, List, Iterable, Tuple
import pickle
import numpy as np


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


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
            return ref_d, None
        elif cand and not ref:
            return None, data
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
            # ref_d = os.path.join('..', 'data_augmentation/data/pheme_annotations/{}-all-rnr-threads.csv'.format(event))
            ref_d = os.path.join('..', 'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
            # ref_d = os.path.join('..', 'data_augmentation/data/ref/{}.csv'.format(event))
            print(os.path.abspath(ref_d))
            ref_d = load_csv(ref_d)
            ref_d = ref_d[['text']]
            # ref_d = ref_d[['text', 'label']]
            # ref_d = ref_d[ref_d['label']=='0']
            ref_d.dropna(inplace=True)
            ref_d.reset_index(inplace=True, drop=True)
            print("Number of original references: ", len(ref_d))

        if cand:
            # data = os.path.join('..', 'data_augmentation/data_hydrator/saved_data/source-tweets/{}/input-cand.pickle'.format(event))
            data = os.path.join('..', 'data_augmentation/data_hydrator/saved_data/source-tweets/{}/input-cand-user.pickle'.format(event))
            print("cand data filename ", os.path.basename(data))
            with open(os.path.join(data), 'rb') as tf:
                data = pickle.load(tf)
                tf.close()
            print("Number of original candidates: ", len(data))
            print("")

        if ref and cand:
            return ref_d, data
        elif ref and not cand:
            return ref_d, None
        elif cand and not ref:
            return None, data
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
        ref_d, data = load_data(name=name, event=event, cand = cand, ref= ref)
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
            new_ref.reset_index(inplace=True, drop=True)

            with open(os.path.join(outfile, 'input-ref-processed.pickle'), 'wb') as tf:
                pickle.dump(new_ref, tf)
                tf.close()

        if cand:
            tokenised_cand, cand_blank = preprocess_tweets(data['text'])
            data['processed_text'] = tokenised_cand
            new_cand = data.drop(cand_blank)
            assert len(new_cand)+len(cand_blank) == len(data)
            new_cand.reset_index(inplace=True, drop=True)

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
        with open(os.path.join(outpath, 'input-{}-processed_11.pickle'.format(t)), 'rb') as tf:
            df = pickle.load(tf)
            tf.close()
        print(len(df))
        pp(list(df))
        print(df.head())

        with open(os.path.join(outpath, 'elmo_{}_input_11.txt'.format(t)), 'w') as f:
            for i, row in df.iterrows():
                f.write(str(row['processed_text']))
                f.write('\n')
            f.close()

        with open(os.path.join(outpath, 'elmo_{}_input_11.txt'.format(t)), 'r') as f:
            lines = f.read().splitlines()
            print(len(lines))
            assert len(lines) == len(df)
            # for i, line in enumerate(f):
            # for i, line in enumerate(lines):
            #     x = df.loc[i, 'processed_text']
            #     if x != line:
            #         raise SystemExit


def add_user_info(event: str):
    """
    Add user infor (screen name and id) for collecting contexts
    Should be skipped when input-cand.pickle contains users' screen names and ids (see /src/source-tweets/prepare_candidates.py)
    :param event:
    :return:
    """

    user_data = os.path.join('..',
                        'data_augmentation/data_hydrator/saved_data/source-tweets/{}/input-cand-user.pickle'.format(
                            event))
    with open(os.path.join(user_data), 'rb') as tf:
        user_data = pickle.load(tf)
        tf.close()
    print("Number of original candidates with user info: ", len(user_data))
    print(list(user_data))
    data = os.path.abspath(
        os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}/input-cand-processed.pickle'.format(event)))

    with open(os.path.join(data), 'rb') as tf:
        data = pickle.load(tf)
        tf.close()
    print("Number of processed candidates w/o user info: ", len(data))
    print(list(data))
    data.set_index('index', inplace=True)
    print(data[10000:50000])
    new_data = user_data[user_data.index.isin(data.index)]

    print(new_data[10000:50000])
    assert len(new_data) == len(data)
    if not data[['id', 'text']].equals(new_data[['id', 'text']]):
        raise AssertionError

    final_df = pd.concat([new_data, data[['processed_text']]], axis=1)
    final_df.reset_index(inplace=True, drop=True)
    print(final_df[10000:50000])
    outfile = os.path.abspath(
        os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
    print(outfile)
    with open(os.path.join(outfile, 'input-cand-user-processed.pickle'), 'wb') as tf:
        pickle.dump(final_df, tf)
        tf.close()

def split_elmo_input():
    """
    Due to resource restriction of Sharc,  elmo input files should be separated when jobs are killed
    :return:
    """
    event = 'ferguson'
    outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
    print(outpath)
    t = 'cand'
    batch_size = 788900
    with open(os.path.join(outpath, 'elmo_{}_input.txt'.format(t)), 'r') as f:
        lines = f.read().splitlines()
        lines1 = lines[:3155558]
        print(len(lines1))

        for i in range(4):
            print(i * batch_size, (i + 1) * batch_size)
            batch = lines1[i * batch_size: (i + 1) * batch_size]
            print(batch[0])
            print(batch[-1])
            print(len(batch))
            print("")

            with open(os.path.join(outpath, 'elmo_cand_input_part1-{}.txt'.format(i)), 'w') as f:
                for l in batch:
                    f.write(l)
                    f.write('\n')
                f.close()


def main():
    event = 'charliehebdo'
    # preprocess_main(name='augmented', event=event, cand=True, ref=True)
    outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
    prepare_input(outpath=outpath, event=event, ref=False, cand=True)


if __name__ == '__main__':
    main()
    # add_user_info(event='sydneysiege')
    # split_elmo_input()

# event='bostonbombings'
# #
# data = os.path.join('..',
#                     'data_augmentation/data_hydrator/file-embed-input/{}/input-cand-user-processed.pickle'.format(event))
# with open(os.path.join(data), 'rb') as tf:
#     data = pickle.load(tf)
#     tf.close()
# print("Number of processed candidates: ", len(data))
#
#
# data = os.path.join('..',
#                     'data_augmentation/data_hydrator/file-embed-input/{}/input-ref-processed.pickle'.format(event))
# with open(os.path.join(data), 'rb') as tf:
#     data = pickle.load(tf)
#     tf.close()
# print("Number of processed ref: ", len(data))
#
# outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
# print(outpath)
# t = 'cand'
# with open(os.path.join(outpath, 'elmo_{}_input.txt'.format(t)), 'r') as f:
#     lines = f.read().splitlines()
#     print(len(lines))
#     f.close()
#
# t = 'ref'
# with open(os.path.join(outpath, 'elmo_{}_input.txt'.format(t)), 'r') as f:
#     lines = f.read().splitlines()
#     print(len(lines))
#     f.close()
#
# event='ferguson'
# i=2
# outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data_hydrator/file-embed-input/{}'.format(event)))
# with open(os.path.join(outpath, 'elmo_cand_input_part1-{}.txt'.format(i)), 'r') as f:
#     lines = f.read().splitlines()
#     print(len(lines))
#     f.close()
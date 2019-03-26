from semeval.semeval_data_processor import load_csv
import os
import pandas as pd
from credbankprocessor import preprocessing_tweet_text
from pprint import pprint as pp

def load_data(event, name='pheme'):
    if name=='pheme':
        ref = os.path.join('..',
                           'data_augmentation/data/pheme_rumour_references/{}.csv'.format(event))
        cand = os.path.join('..',
                            'data_augmentation/data/candidates/{}.csv'.format(event))
        # ref = os.path.join(ref_path, '{}.csv'.format(event))
        # cand = os.path.join(data_path, '{}.csv'.format(event))
        ref = load_csv(ref)
        data = load_csv(cand)
        print(ref)
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


def prepare_input(name, event):
    """
    1. Generate input files for ELMO embed_file method; each line contains a sentence tokenized by whitespace.
    :return: text files (input to ELMo)
    """
    if name == 'semeval':
        data = load_data(name=name)
        outfile = os.path.join(os.path.dirname(__file__), '..', 'data/semeval2015/file-embed-input')
        tokenised_tweet_cand = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet1'].values))
        tokenised_tweet_ref = list(map(lambda x: " ".join(literal_eval(x)), data['processed_tweet2'].values))

    elif name=='pheme':
        ref, data = load_data(name=name, event=event)
        outfile = os.path.abspath(
            os.path.join('..', 'data_augmentation/data/file-embed-input/{}'.format(event)))

        processed_cand = list(map(lambda x: preprocessing_tweet_text(x), data['text'].values))
        processed_ref = list(map(lambda x: preprocessing_tweet_text(x), ref['text'].values))
        tokenised_tweet_cand = list(map(lambda x: " ".join(x), processed_cand))
        tokenised_tweet_ref = list(map(lambda x: " ".join(x), processed_ref))

    outfile = os.path.abspath(outfile)
    os.makedirs(outfile, exist_ok=True)
    print(outfile)

    for t in tokenised_tweet1:
        with open(os.path.join(outfile, 'input-cand.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
    f.close()

    for t in tokenised_tweet2:
        with open(os.path.join(outfile, 'input-ref.txt'), 'a') as f:
            f.write(t)
            f.write("\n")
    f.close()

    print("Done")

def empty_indices(event, t='candidates', action='save'):
    """
    ELMo embed_file raises error if there are empty strings --> remove
    :param: event
    :param: t: 'candidates' or 'ref'
    :param: action: whether to save or load empty strings' indices
    :return:
    """
    outpath = os.path.abspath(os.path.join('..', 'data_augmentation/data/{}'.format(t)))
    if action == 'save':
        infile = os.path.join('..',
                              'data_augmentation/data/file-embed-input/{}/input-ref.txt'.format(event))
        input_file = open(infile, 'r')
        sentences = [line.strip() for line in input_file.readlines()]
        blank_lines = [i for (i, line) in enumerate(sentences) if line == ""]
        print("Total number of {} tweets: {} ".format(t, len(sentences)))
        print("Number of empty strings: ", len(blank_lines))

        print("")
        print("Saving the indices of empty strings...")
        with open(os.path.join(outpath, '{}_{}_empty_index.pickle'.format(event, t)), 'wb') as f:
            pickle.dump(blank_lines, f)

    elif action =='load':
        print("Loading the indices of empty strings....")
        with open(os.path.join(outpath, '{}_{}_empty_index.pickle'.format(event, t)), 'rb') as f:
            ids = pickle.load(f)
        return ids


def remove_empty_strings(event, indices):
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
    prepare_input(name='pheme', event='sydneysiege')
    # empty_indices(event='sydneysiege', t='ref', action='save')
    # remove_empty_lines_from_input(event='sydneysiege', check_empty_indices(event='sydneysiege', action='load', t='candidates'))

main()
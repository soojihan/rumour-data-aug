import re
from gensim.utils import deaccent
import string
from nltk.tokenize import TweetTokenizer
import os
import pandas as pd
import csv
import unicodedata


def export_credbank_trainset(dataset_dir):
    """
    export credbank as trainset for ELMo fine-tune

    :param dataset_dir:
    :return:
    """

    dataset_files = load_all_files_path(dataset_dir)
    export_file_path = os.path.join(dataset_dir, "credbank_train_corpus.txt")
    all_tweets_size = 0
    all_dedup_tweets_size = 0
    vocabulary = set()
    with open(export_file_path, mode='w', encoding='utf-8') as outputfile:
        for credbank_corpus_batch_i in dataset_files:
            print("loading, preprocessing and export corpus in batch from [%s]" % credbank_corpus_batch_i)
            tweet_corpus_batch_i = load_tweets_from_credbank_csv(credbank_corpus_batch_i)
            all_collected_tweets_from_batch_i = list(tweet_corpus_batch_i)
            all_tweets_size_batch_i = len(all_collected_tweets_from_batch_i)
            print("totals tweets collected from current batch: [%s]" % all_tweets_size_batch_i)
            all_tweets_size += all_tweets_size_batch_i
            dedup_tweet_corpus_batch_i = set(all_collected_tweets_from_batch_i)
            all_dedup_tweets_size_batch_i = len(dedup_tweet_corpus_batch_i)
            print("total deduplicated tweets from current batch: [%s]" % all_dedup_tweets_size_batch_i)
            all_dedup_tweets_size += all_dedup_tweets_size_batch_i
            for tweet_text in dedup_tweet_corpus_batch_i:
                outputfile.write("%s\n" % tweet_text)
            print("done.")
        print("all tweet sentences: ", all_tweets_size) # 77954446
        print("all deduplicated tweet sentences: ", all_dedup_tweets_size) # 6157180
        print("all complete.")


def generate_train_held_out_set(train_corpus_path):
    """
    generate small held-out set for testing language model perplexity (compare perplexity before and after fine-tune)
    :param train_corpus_path:
    :return:
    """
    from sklearn.model_selection import ShuffleSplit
    with open(train_corpus_path, mode='r', encoding='utf-8') as train_file:
        train_set = train_file.readlines()

    # with test_size=0.0002, we will have 1232 tweets in held-out set

    # total number of train set:  6155948
    # total number of held set:  1232
    rs = ShuffleSplit(n_splits=1, random_state=0, test_size=0.0002, train_size=None)
    splitted_sets = list(rs.split(train_set))
    shuffled_train_set = splitted_sets[0][0]
    shuffled_held_set = splitted_sets[0][1]

    print("total number of train set: ", len(shuffled_train_set))
    print("total number of held set: ", len(shuffled_held_set))
    #print("train set: ", shuffled_train_set)
    #print("held set: ", shuffled_held_set)

    train_data_dir = os.path.dirname(train_corpus_path)
    shuffled_train_set_path = os.path.join(train_data_dir, "shuffled_credbank_train_corpus.txt")
    shuffled_held_set_path = os.path.join(train_data_dir, "shuffled_credbank_held_corpus.txt")

    with open(shuffled_train_set_path, mode='w', encoding='utf-8') as outputfile:
        for shuffled_train_indice in shuffled_train_set:
            outputfile.write("%s" % train_set[shuffled_train_indice])

    with open(shuffled_held_set_path, mode='w', encoding='utf-8') as outputfile:
        for shuffled_held_indice in shuffled_held_set:
            outputfile.write("%s" % train_set[shuffled_held_indice])

    print("shuffled train and held-out set are generated and exported.")


def _load_matrix_from_csv(fname,start_col_index, end_col_index, delimiter=',', encoding='utf-8', header=None):
    """
    load gs terms (one term per line) from "csv" txt file
    :param fname:
    :param start_col_index:
    :param end_col_index:
    :param encoding:
    :param header default as None, header=0 denotes the first line of data
    :return:
    """
    df = pd.read_csv(fname, header=header, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, usecols=range(start_col_index, end_col_index), lineterminator='\n', encoding=encoding).as_matrix()
    return df


def load_tweets_from_credbank_csv(credbank_dataset_path):
    """
    load tweet text from 9th column

    preprocess the text for fine-tuning ELMo model

    "Each file contains pre-tokenized and white space separated text, one sentence per line. Don't include the <S> or </S> tokens in your training data."
    https://github.com/allenai/bilm-tf

    :param credbank_dataset_path:
    :return:
    """
    df = _load_matrix_from_csv(credbank_dataset_path, delimiter="\t", start_col_index=8, end_col_index=10)
    for tweet_row in df[:]:
        tweet_text = tweet_row[0]
        # print(type(tweet_text))
        if str(tweet_text) != 'nan':
            # preprocessing_tweets()
            norm_tweet = preprocessing_tweet_text(tweet_text)
            yield " ".join(norm_tweet)


def load_all_files_path(dataset_dir):
    all_files = []
    for file in os.listdir(dataset_dir):
        all_files.append(os.path.join(dataset_dir, file))

    return all_files


def _run_strip_accents(self, text):
    """Strips accents from a piece of text.
    Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    # Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py

    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    # Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def preprocessing_tweet_text(tweet_text):
    """
    Neural Language Model like ELMo does not need much normalisation. Pre-trained ELMo model only need pre-tokenised text.

    :param tweet_text:
    :return:
    """
    if not isinstance(tweet_text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    norm_tweet = tweet_text.lower()
    # remove retweets
    norm_tweet = re.sub('rt @?[a-zA-Z0-9_]+:?', '', norm_tweet)
    # remove URL
    norm_tweet = re.sub(r"http\S+", "", norm_tweet)
    # remove pic URL
    norm_tweet = re.sub(r"pic.twitter.com\S+", "", norm_tweet)
    # remove user mentions
    norm_tweet = re.sub(r"(?:\@|https?\://)\S+", "", norm_tweet)
    # remove punctuations:
    # norm_tweet = re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', repl='', string=norm_tweet).strip()
    # deaccent
    norm_tweet = deaccent(norm_tweet)

    tknzr = TweetTokenizer()
    tokenised_norm_tweet = tknzr.tokenize(norm_tweet)

    # https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

    # Set the minimum number of tokens to be considered
    if len(tokenised_norm_tweet) < 4:
        return []

    num_unique_terms = len(set(tokenised_norm_tweet))

    # Set the minimum unique number of tokens to be considered (optional)
    if num_unique_terms < 2:
        return []

    return tokenised_norm_tweet


def corpus_statistics():
    #train_corpus_path = "/userstore/jieg/credbank/corpus/credbank_train_corpus.txt"
    train_corpus_path = "C:\\Data\\credbank\\tweets_corpus\\shuffled_credbank_held_corpus.txt"
    with open(train_corpus_path, mode='r', encoding='utf-8') as file:
        train_corpus = file.readlines()

    from nltk.tokenize.regexp import WhitespaceTokenizer
    whitespace_tokenize = WhitespaceTokenizer().tokenize
    corpus_size = 0
    for tweet in train_corpus:
        tokens = whitespace_tokenize(tweet)
        corpus_size += len(tokens)

    print("all words (corpus size): ", corpus_size)

    from sklearn.feature_extraction.text import CountVectorizer

    #extract tokens
    text_vectorizer = CountVectorizer(analyzer='word', tokenizer=WhitespaceTokenizer().tokenize, ngram_range=(1, 1), min_df=1)
    X = text_vectorizer.fit_transform(train_corpus)
    # Vocabulary
    vocab = list(text_vectorizer.get_feature_names())
    print("vocabulary size: ", len(vocab)) # 913611
    counts = X.sum(axis=0).A1

    from collections import Counter
    freq_distribution = Counter(dict(zip(vocab, counts)))

    print("top N frequent words: ", freq_distribution.most_common(10))


def test():
    tweet_1 = "RT @TheManilaTimes: Cheers, tears welcome Pope Francis - The Manila Times OnlineThe Manila Times Online http://www.manilatimes.net/cheers-tears-welcome-pope-francis/155612/ …	3:31 am - 15 Jan 2015"
    pre_tweet_1 = preprocessing_tweet_text(tweet_1)
    print(pre_tweet_1)

    tweet_2 = "Welcome to the Philippines Pope Francis @Pontifex Pray for the Philippines & the entire world.	5:19 pm - 15 Jan 2015"
    pre_tweet_2 = preprocessing_tweet_text(tweet_2)
    print(pre_tweet_2)

    tweet_3 = "Retweet if you're proud Filipino! \"Welcome to the Philippines Pope Francis\" http://bit.ly/150Zqcq  ۞| http://bit.ly/1INBcie 	3:02 pm - 15 Jan 2015"
    pre_tweet_3 = preprocessing_tweet_text(tweet_3)
    print(pre_tweet_3)

    tweet_4 = "Why Lambert, Lovren and Lallana have struggled at Liverpool http://dlvr.it/8gqKRv  @PLNewsNow"
    pre_tweet_4 = preprocessing_tweet_text(tweet_4)
    print(pre_tweet_4)

# test()


#tweet_corpus = load_tweets_from_credbank_csv("C:\\Data\\credbank\\credbank_xai.csv")
#all_tweets = set(list(tweet_corpus))
#for tweet_text in all_tweets:
#    print(tweet_text)
#print("all tweets: ", len(all_tweets))

# export_credbank_trainset("C:\\Data\\credbank\\tweets_corpus")

# corpus_statistics()

# generate_train_held_out_set("/fastdata/ac1jgx/credbank/train/credbank_train_corpus.txt")
#generate_train_held_out_set("C:\\Data\\credbank\\credbank_train_corpus.txt")
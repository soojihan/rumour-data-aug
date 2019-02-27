import string
import unicodedata
import logging
import re

# from .snowball import SnowballStemmer
# from .stopwords__ import get_stopwords_by_language
# from ..syntactic_unit import SyntacticUnit
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk import pos_tag
from replacers import RepeatReplacer
# import CMUTweetTagger
from pprint import pprint as pp

np.random.seed(2018)

# Utility functions adapted from Gensim v0.10.0:
# https://github.com/RaRe-Technologies/gensim/blob/0.10.0/gensim/utils.py
# https://github.com/RaRe-Technologies/gensim/blob/0.10.0/gensim/parsing/preprocessing.py


SEPARATOR = r"@"
RE_SENTENCE = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)')
AB_SENIOR = re.compile("([A-Z][a-z]{1,2}\.)\s(\w)")
AB_ACRONYM = re.compile("(\.[a-zA-Z]\.)\s(\w)")
AB_ACRONYM_LETTERS = re.compile("([a-zA-Z])\.([a-zA-Z])\.")
UNDO_AB_SENIOR = re.compile("([A-Z][a-z]{1,2}\.)" + SEPARATOR + "(\w)")
UNDO_AB_ACRONYM = re.compile("(\.[a-zA-Z]\.)" + SEPARATOR + "(\w)")

LANGUAGES = {"english"}

STEMMER = PorterStemmer() # Define a stemmer
# STEMMER = SnowballStemmer('english')
LEMMATISER = WordNetLemmatizer() # Define a lemmtiser
REPEAT_REPLACER = RepeatReplacer()

def set_stemmer_language(language):
    global STEMMER
    if not language in LANGUAGES:
        raise ValueError("Valid languages are danish, dutch, english, finnish," +
                         " french, german, hungarian, italian, norwegian, porter, portuguese," +
                         "romanian, russian, spanish, swedish")
    STEMMER = PorterStemmer(language)

def set_stopwords_by_language(language): # Setting a stopword list
    # words = get_stopwords_by_language(language)
    # gensim_stopwords = list(STOPWORDS)
    punct = list(string.punctuation)
    worldwide_stopwords = []
    for lang in LANGUAGES:
        worldwide_stopwords.append(stopwords.words("{}".format(lang)))
    worldwide_stopwords = [x for sublist in worldwide_stopwords for x in sublist]
    stopword_list = worldwide_stopwords +  punct + \
                    ["hurricane", "tropical", "harvey", 'lol', 'v', 'rt', 'via', "...", "could", "would", "should",
                     " - ", " v ", " vs ", "storm", "#hurricaneharvey", "hurricaneharvey", "u", "w", "s", "n", "boston",
                    "marathon", "bostonmarathon", "prayforboston", "prayforottawa", "ottawa", "mahrez", "riyad", "election2012", "ferguson",
                     "sydney", "sydneysiege", "election", "vote", "oh", "ooh", "fuck", "poll"]
                    # "marathon", "bostonmarathon", "prayforboston", "prayforottawa", "ottawa"] + words.split()

    STOPWORDS = frozenset(w for w in stopword_list if w)
    return STOPWORDS


def init_textcleanner(language):
    # set_stemmer_language(language)
    set_stopwords_by_language(language)


def split_sentences(text):
    processed = replace_abbreviations(text)
    return [undo_replacement(sentence) for sentence in get_sentences(processed)]


def replace_abbreviations(text):
    return replace_with_separator(text, SEPARATOR, [AB_SENIOR, AB_ACRONYM])


def undo_replacement(sentence):
    return replace_with_separator(sentence, r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])


def replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def get_sentences(text):
    for match in RE_SENTENCE.finditer(text):
        yield match.group()


# Taken from Gensim
RE_PUNCT = re.compile('([%s])+' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'), re.UNICODE) # remove '_' from punctuation set (because of mentions)
def strip_punctuation(s):
    return RE_PUNCT.sub(" ", s)


# Taken from Gensim
RE_NUMERIC = re.compile(r"^[<>{}\"/|;:.,~!?@#$%^=&*\\]*$", re.UNICODE)  # strings composed of numbers and/or special symbols

def strip_numeric(s):
    return RE_NUMERIC.sub("", s)


def remove_stopwords(sentence):
    STOPWORDS = set_stopwords_by_language("english")
    return " ".join(w for w in sentence.split() if w not in STOPWORDS)


RE_SINGLE_ALPHABET = re.compile(r'(?i)\b[a-z]\b', re.UNICODE)


def strip_single_alph(s):
    return RE_SINGLE_ALPHABET.sub("", s)


def stem_sentence(sentence):
    word_stems = [STEMMER.stem(word) for word in sentence.split()]
    # word_stems = STEMMER.stem(sentence)
    return " ".join(word_stems)


def lemmatize_sentence(sentence):
    # word_lemmas = [LEMMATISER.lemmatize(word) for word in sentence.split()]
    word_lemmas_verb = [LEMMATISER.lemmatize(word, pos='v') for word in sentence.split()]
    word_lemmas = [LEMMATISER.lemmatize(word) for word in word_lemmas_verb]
    # print("word lemms ", word_lemmas)
    return " ".join(word_lemmas)

def lemmatize_stemming(sentence):
    processed_text = [STEMMER.stem(LEMMATISER.lemmatize(word, pos='v')) for word in sentence.split()]
    return " ".join(processed_text)

def apply_filters(sentence, filters):
    for f in filters:
        sentence = f(sentence)
    return sentence


def filter_words(sentences):
    # Choose whether to lemmatize or stemmize
    filters = [lambda x: x.lower(), strip_numeric, strip_punctuation, remove_stopwords,
               lemmatize_sentence] # tokenpattern here filters single

    apply_filters_to_token = lambda token: apply_filters(token, filters)
    return list(map(apply_filters_to_token, sentences))


# Taken from Gensim
def deaccent(text):
    """
    Remove accentuation from the given string.
    """
    norm = unicodedata.normalize("NFD", text)
    result = "".join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def tokenpattern(text):
    """"
    remove single alphabet # problem: remove 1 but save 10 > solved
    """
    pattern = r"([0-9]+|[a-zA-Z_]\w+|\d\w+)" # sinlge number | alphabet+alphanumeric character | single numer + alphanumeric character
    token_pattern = re.compile(pattern)
    match = token_pattern.findall(text)
    return " ".join(match)


# Taken from Gensim
PAT_ALPHABETIC = re.compile('(\w+)', re.UNICODE)  # leave digits @caribbean360 > caribbean360

def tokenize(text, lowercase=False, deacc=False, token_pattern=True):
    """
    Iteratively yield tokens as unicode strings, optionally also lowercasing them
    and removing accent marks.
    """
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    if token_pattern:
        text = tokenpattern(text)

    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()



def tokenize_by_word(text, language="english", deacc=True):
    """ Actual sentence preprocessing """
    init_textcleanner(language)
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    tokeniser = TweetTokenizer()
    tweet_tokenized = tokeniser.tokenize(text_without_acronyms)
    joined_tokens = " ".join(word for word in tweet_tokenized if not word.startswith('@')) # remove mentions from tweets#
    joined_tokens = re.sub('#\S+', '', joined_tokens)  # remove hashtags
    tokenized = list(tokenize(joined_tokens, lowercase=False, deacc=deacc))
    filtered_words = filter_words(tokenized)
    filtered_words = [REPEAT_REPLACER.replace(x) for x in filtered_words] # 'loooooooove' to 'love'
    return filtered_words



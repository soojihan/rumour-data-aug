import twint
from glob import glob
import pandas as pd
from pprint import pprint as pp
import os
# import tweepy
import json
from twarc import Twarc
import argparse
import time
import pickle
from ast import literal_eval
from optparse import OptionParser
from sys import platform

"""
Collect tweets for new events using Twint
https://github.com/twintproject/twint
"""

pd.set_option('display.expand_frame_repr', False)


class twitter_configuration:
    def __init__(self):
        self.consumer_key='LAEnCQo2qD4SXKorY85SpV7Bw'
        self.consumer_secret='N6rZLSoVihWoqoqdjsAuxOuTaZAThAxc9PBjsP0Nkrbh6X8H71'
        self.access_token='803794891000205314-Fve4yFzgOW8SeuJlapIOtQ0HPYrUOOP'
        self.access_token_secret='enplY3OQzPPQA2ooMeVyfUPvJxzGEwMoSzcPx9lZc26Yq'

    def config(self, lib_type='twarc'):
        if lib_type == 'tweepy':
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            c = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        elif lib_type == 'twint':
            c = twint.Config()
        elif lib_type == 'twarc':
            c = Twarc(self.consumer_key, self.consumer_secret, self.access_token, self.access_token_secret)
            c = c.config()
        return c

t = twitter_configuration()
c = t.config(lib_type='twint')

def search_tweets(keyword, since_t, until_t, output_path):
    c.Since = since_t
    c.Until = until_t
    # c.Limit = 2000
    c.Search = keyword
    c.Store_json = True
    c.User_full = True
    c.Lang = 'en'
    c.Output = os.path.join(output_path, "{}.json".format(keyword))
    c.Format = "Tweet id: {id} | Timezone: {timezone} | Time: {time} | Date: {date} | Tweet: {tweet} | Replies: {replies}"
    twint.run.Search(c)

def collect_new_event(keywords, output_path, since, until):
    start = time.time()

    os.makedirs(output_path, exist_ok=True)

    for i, keyword in enumerate(keywords):
        print("")
        print(i, keyword)
        if os.path.exists(os.path.join(output_path, "{}.json".format(keyword))):
            print("already exists...")
            continue
        else:
            # tweets = search_tweets(keyword, since_t='2019-03-15', until_t='2019-03-17', output_path=output_path)
            tweets = search_tweets(keyword=keyword, since_t=since, until_t=until, output_path=output_path)
            pp(tweets)
    end = time.time()
    print("{}".format(time.strftime("%H:%M:%S", time.gmtime(end-start))))

def main():
    if platform == 'linux':
        parser = OptionParser()
        parser.add_option(
            '--keywords', dest='keywords', default="['#ChristchurchTerrorAttack']",
            help='List of keywords for searching tweets (list): default=%default')
        parser.add_option(
            '--output', dest='output', default='/path/to/store/output',
            help='Path to save downloaded tweets (string) : default=%default')
        parser.add_option(
            '--since', dest='since', default='2019-03-26',
            help='Filter tweets sent since date (string): default=%default')
        parser.add_option(
            '--until', dest='until', default='2019-03-27',
            help='Filter tweets sent until date (string): default=%default')

        (options, args) = parser.parse_args()
        keywords = options.keywords
        keywords = literal_eval(keywords)
        print("keywords: {}".format(keywords))
        output_path = options.output
        since = str(options.since)
        until = str(options.until)

    elif platform =='darwin':
        keywords = ['#ChristchurchTerrorAttack', '#christchurch', '#NewZealandMosqueAttack', '#NewZealandShooting', 'christchurch', 'new zealand']
        output_path = os.path.join('..', 'data_augmentation/christichurch-shooting')
        since = '2019-03-15'
        until = '2019-03-17'
    collect_new_event(keywords, output_path, since, until)


main()




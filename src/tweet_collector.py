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

"""
Retreive the top 100 retweets of each context tweet in the Pheme data
"""

pd.set_option('display.expand_frame_repr', False)

parser = argparse.ArgumentParser()
parser.add_argument('--event', help='the name of event')
parser.add_argument('--df_path', help='path to tweet ids which will be downloaded')
# parser.add_argument('--save_path', help='path to save downloaded tweets')

args = parser.parse_args()
event = args.event
df_path = args.df_path
print(df_path)
# save_path = args.save_path

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
    c.Limit = 2000
    c.Search = keyword
    c.Store_json = True
    c.User_full = True
    c.Get_replies = True
    c.Replies = True
    c.Lang = 'en'
    c.Output = os.path.join(output_path, "{}.json".format(keyword))
    c.Format = "Tweet id: {id} | Timezone: {timezone} | Time: {time} | Date: {date} | Tweet: {tweet} | replies: {replies}"
    twint.run.Search(c)

def collect_new_event():
    start = time.time()
    keywords = ['#ChristchurchTerrorAttack', '#christchurch', '#NewZealandMosqueAttack', '#NewZealandShooting', 'christchurch', 'new zealand']
    output_path = os.path.join('..', 'data_augmentation/christichurch-shooting')
    os.makedirs(output_path, exist_ok=True)

    for i, keyword in enumerate(keywords):
        print("")
        print(i, keyword)
        if os.path.exists(os.path.join(output_path, "{}.csv".format(keyword))):
            print("already exists...")
            continue
        else:
            tweets = search_tweets(keyword, since_t='2019-03-15', until_t='2019-03-17', output_path=output_path)
            pp(tweets)
    end = time.time()
    print("{}".format(time.strftime("%H:%M:%S", time.gmtime(end-start))))

collect_new_event()




import twint
import glob
import pandas as pd
from pprint import pprint as pp
import os
import shutil
import tweepy
import json

"""
Rearrange retweets collected using 'retweet_collector.py' to be compatible with the original PHEME dataset
"""
# consumer_key =  "yTaqhLcxY2SJTcJSbe9CTlfWI"
# consumer_secret = "au6Mj1lMGzlnDkkqcZrCoowzylJ0yxsAlmVLXRKDwpUkU7uXUu"
# access_token = "803794891000205314-bs7wteyEIAxuETlcxkjOyvtrCphiKsD"
# access_token_secret = "1BGjkLMRxDrNy4oqxqxAIEMLYQrBEZzcZAvvcClozizPX"

consumer_key='aH2d3A8Rnp6gWDjWWUpf6VWRC'
consumer_secret='2vgPHohvytNYhgLcSRRargR3jIR54TG8iiZnNqldd1JinSPDBJ'
access_token='803794891000205314-kUUntk0hGeUZuEW3eHgTHxCKeM5rwNq'
access_token_secret='zO4iovo63I0ACWDBhLgEdhVXhm9kA60eCT2Pm97b9jNfN'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def lookup_(tweet_ids, api, batch_size=100):
    eng_tweet = []
    tweet_count = len(tweet_ids)

    while True:
         try:
             for i in range(int(tweet_count / batch_size) + 1):

                batch_ids = tweet_ids[i * batch_size : min((i + 1) * batch_size, tweet_count)]
                print('getting users batch {} of size {}'.format(i, len(batch_ids)))
                output = api.statuses_lookup(batch_ids)

                for tw in output:
                    # print(tw._json['id_str'], tw._json['lang'])
                    if tw._json['lang'] == 'en':
                        with open(output_path, 'a') as outfile:
                            outfile.write(tw._json['id_str'] + '\n')
                # eng_tweet.extend(api.statuses_lookup(batch_ids))

                print("")
         except tweepy.TweepError as e:
             print('Something went wrong, quitting...', e)
             # time.sleep(15 * 60)
         return eng_tweet


def reformat_retweets():
    """
    Change the format of retweets in the PHEME dataset
    retweets.json contains all retweets of each source tweet.
    We follow the PHEME structure /event/rumour_category/source_tweet_id/retweets/retweet_id/retweet_id.json

    :return: JSON containing a single retweet
    """
    input_files = glob.glob(os.path.join('..', 'data/train/missingids/*/retweets.json'))

    outpath = glob.glob(os.path.join('..', '..', '..', 'Data/all-rnr-annotated-threads/*/*/*'))
    split_path = list(map(lambda x: x.split('/'), outpath))
    ids = [x[-1] for x in split_path]
    print(len(ids))

    for f in input_files:
        source_tweet_id = f.split('/')[4]
        print(source_tweet_id)

        with open(f, 'r') as infile:
            retweets = json.load(infile)
            for rt in retweets:
                rt_id = rt['id']
                rt.pop('retweeted_status') # retweets have the same content (original source tweet)
                if source_tweet_id in ids:
                    # outfile = os.path.join('..', '..','..','Data/all-rnr-annotated-threads/*/*/{}/retweets/{}'.format(source_tweet_id, rt_id))
                    outfile = glob.glob(os.path.join('..', '..','..','Data/all-rnr-annotated-threads/*/*/*'))
                    pp(outfile)
                    for outpath in outfile:
                        if outpath.split('/')[-1] == source_tweet_id:
                            json_path = os.path.join(outpath,'retweets')
                            print("json path ", json_path)
                            os.makedirs(json_path, exist_ok=True)
                            with open(os.path.join(json_path, '{}.json'.format(rt_id)), 'w') as f:
                                json.dump(rt, f)
                                # raise SystemExit


                # os.makedirs(outpath, exist_ok=True)
                # with open(os.path.join(outpath, '{}.json'.format(rt_id)), 'w') as f:
                #     json.dump(rt, f)
#
reformat_retweets()

## rename
# outfile = glob.glob(os.path.join('..', '..','..','Data/all-rnr-annotated-threads/*/*/*'))
# for outpath in outfile:
#     # print(outpath)
#     # if outpath.split('/')[-1] == source_tweet_id:
#     json_path = os.path.join(outpath, 'rewteets')
#     if os.path.exists(json_path):
#             os.rename(json_path, os.path.join(outpath, 'retweets'))


    #     files = os.listdir(json_path)

        # for f in files:
        #     f_path = os.path.join(json_path, f)
        #     print(f_path)
        #     print(os.path.isdir(f_path))
        #     if os.path.isdir(f_path):
        #         os.rmdir(f_path)
            # json_f = glob.glob(os.path.join(f_path, '*.json'))
            # print(json_f)
            # shutil.move(json_f[0], json_path)

# source =
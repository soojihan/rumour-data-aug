import twint
from glob import glob
import pandas as pd
from pprint import pprint as pp
import os
# import tweepy
# import twint
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
        return c

c = twitter_configuration()
t = c.config()
print(t)



# df_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
#                                        # '..', 'data_augmentation/data/augmented_data_annotation/boston'))
#                                        '..', 'data_augmentation/data/file-embed-output/boston/results_p9003<0.3'))

def load_data(path):
    """
    Load merged training pheme data
    :param path: path to files
    :return: merged dataframe
    """
    files = glob.glob(os.path.join(path, '*.csv'))
    pp(files)
    final_df = pd.DataFrame()
    for file in files:
        with open(file, 'r') as f:
            df = pd.read_csv(f)
            # df.drop(['Unnamed: 0'], axis=1, inplace=True)
            final_df = final_df.append(df)
    return final_df

def id_extractor(df_path):
    """
    Extract ids and save as txt file (for Twarc tweet collector)
    :param df_path:
    :return:
    """
    df = load_data(df_path)
    df.drop_duplicates(subset=['id'], inplace=True)
    ids = list(df['id'].values)
    print(ids)
    # outfile = os.path.join(df_path, 'ids.txt')
    outfile = os.path.join(df_path, 'boston-9003-source-ids.txt')
    with open(outfile, 'w') as f:
        for l in ids:
            f.write(str(int(l)))
            f.write('\n')
        f.close()

# id_extractor(df_path)
# raise SystemExit

def tweet_retweet_collector(df_path):
    """
    Download tweets and retweets at the same time
    input: list of ids
    :return:
    """
    print(os.path.exists(os.path.join(df_path, 'boston-9003-source-ids.txt')))
    count = 0
    start = time.time()
    ## Download tweets
    for tweet in t.hydrate(open(os.path.join(df_path, 'boston-9003-source-ids.txt'))):
        id = tweet['id_str']
        save_path_source = os.path.join(df_path, 'hydrates-9003/{}/source-tweets'.format(id))
        save_path_retweets = os.path.join(df_path, 'hydrates-9003/{}/retweets'.format(id))
        if (os.path.exists(os.path.join(save_path_source, '{}.json'.format(id)))) or tweet['lang']!='en':
            print("pass")
            continue
        os.makedirs(save_path_source, exist_ok=True)
        os.makedirs(save_path_retweets, exist_ok=True)
        with open(os.path.join(save_path_source, '{}.json'.format(id)), 'w') as f:
            json.dump(tweet, f, indent=1)
            f.close()


        ## Download retweets
        retweets = t.retweets(id)
        for rt in retweets:
            rt_id = rt['id_str']
            rt.pop('retweeted_status')
            with open(os.path.join(save_path_retweets, '{}.json'.format(rt_id)), 'w') as f:
                json.dump(rt, f, indent=1)
                f.close()
            print(rt)
        count+=1
        if count % 1000 ==0:
            end = time.time()
            print("{} complete, {}".format(count, end-start))

def augmented_data_collector(df_path):
    """
    Download augmented data (replies)
    source tweet ids are required
    :return:
    """
    start = time.time()
    ## Download tweets
    # print(df_path)
    source_ids = os.path.join('..', '..', 'data_augmentation/data/file-embed-output/boston/results_p9003<0.3/boston-9003.csv')
    source_ids = pd.read_csv(source_ids)
    print(list(source_ids))
    for i, row in source_ids.iterrows():
        source_id = str(int(row['id']))
        print("Source id ", source_id)
        label = row['label']
        rtype = 'rumours' if label==1 else 'non-rumours'
        # reply_path = "/mnt/fastdata/acp16sh/data-aug/data_augmentation/data/augmented_data/boston-9003"
        reply_path = "/Users/suzie/Desktop/PhD/Projects/data-aug/data_augmentation/data/augmented_data/boston-9003"
        reply_ids = glob("/Users/suzie/Desktop/PhD/Projects/data-aug/data_augmentation/data/augmented_data/boston-9003/*/{}/*.pickle".format(source_id))
        print(reply_ids)

        if reply_ids:
            reply_ids = reply_ids[0] # replies pickle file
            print(reply_ids)
            with open(reply_ids, 'rb') as f:
                rpl_ids = pickle.load(f)
                print(rpl_ids)


            for tweet in t.hydrate(rpl_ids):
                # print(tweet)
                r_id = tweet['id_str']
                save_path_source = os.path.join(reply_path, '{}/{}/reactions'.format(rtype, source_id))
                print(save_path_source)
                if (os.path.exists(os.path.join(save_path_source, '{}.json'.format(id)))) or tweet['lang']!='en':
                    print("pass")
                    continue
                os.makedirs(save_path_source, exist_ok=True)
                with open(os.path.join(save_path_source, '{}.json'.format(r_id)), 'w') as f:
                    print("dumping...")
                    json.dump(tweet, f, indent=1)
                    f.close()

# augmented_data_collector(df_path)

def retrieve_retweets():
    """
    retrieve the first 100 retweets using Twitter API (twint)

    :return:
    """
    print("Loading data...")
    df = load_data(df_path)
    # df.drop_duplicates(subset=['tweet_id'], inplace=True)
    df.drop_duplicates(subset=['id'], inplace=True)

    pp(list(df))
    print(len(df))
    df.reset_index(inplace=True, drop=True)

    for i, row in df.iterrows():
        user_id = row['user_id']
        tweet_id = row['tweet_id']
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train/missingids/{}'.format(tweet_id))
        print("")
        print(output_path)
        if os.path.exists(output_path):
            print(i, "pass")
            continue
        else:
            try:
                results = api.retweets(tweet_id, count=1000)
                print("{} Tweet id: {}, num of retweets: {}".format(i, tweet_id, len(results)))
                # output_path = '../data/train_test/{}'.format(tweet_id)
                # print(output_path)
                os.makedirs(output_path, exist_ok=True)
                retweets_json = []
                for obj in results:
                    retweets_json.append(obj._json)
                with open(os.path.join(output_path, 'retweets.json'), 'w') as f:
                    json.dump(retweets_json, f, indent=1)

            except tweepy.TweepError as e:
                print('Something went wrong, quitting...', e)

def check_missing_ids():
    missing_ids = pd.DataFrame(columns=list(df))
    for i, row in df.iterrows():
        user_id = row['user_id']
        tweet_id = row['tweet_id']
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train/{}'.format(tweet_id))
        print(output_path)
        if os.path.exists(output_path):
            print(i, "pass")
            continue
        else:
            missing_ids.loc[len(missing_ids)] = row
    print(len(missing_ids))
    print(missing_ids)
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train/missing_ids.csv')
    missing_ids.to_csv(output_path)


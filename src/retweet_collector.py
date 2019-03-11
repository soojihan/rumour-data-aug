import twint
import glob
import pandas as pd
from pprint import pprint as pp
import os
import tweepy
import json
from twarc import Twarc

# consumer_key =  "yTaqhLcxY2SJTcJSbe9CTlfWI"
# consumer_secret = "au6Mj1lMGzlnDkkqcZrCoowzylJ0yxsAlmVLXRKDwpUkU7uXUu"
# access_token = "803794891000205314-bs7wteyEIAxuETlcxkjOyvtrCphiKsD"
# access_token_secret = "1BGjkLMRxDrNy4oqxqxAIEMLYQrBEZzcZAvvcClozizPX"

consumer_key='LAEnCQo2qD4SXKorY85SpV7Bw'
consumer_secret='N6rZLSoVihWoqoqdjsAuxOuTaZAThAxc9PBjsP0Nkrbh6X8H71'
access_token='803794891000205314-Fve4yFzgOW8SeuJlapIOtQ0HPYrUOOP'
access_token_secret='enplY3OQzPPQA2ooMeVyfUPvJxzGEwMoSzcPx9lZc26Yq'

## tweepy configuration
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

## twint configuration
c = twint.Config()
## twarc configuration
t = Twarc((consumer_key, consumer_secret, access_token, access_token_secret))

#
# c.User_id = '1960878613'
#
# c.User_full = False
# c.Format = "Username: {username} | Bio: {bio} | Url: {url}"
#
# # c.Store_csv = True
# c.Output = "none"
#
# twint.run.Followers(c)

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


# df = load_data('../data/pheme-training')
# df = load_data('../data/train')
df_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train')
print("df path ", df_path)
df = load_data(df_path)

df.drop_duplicates(subset=['tweet_id'], inplace=True)

pp(list(df))
print(len(df))
df.reset_index(inplace=True, drop=True)

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

def retrieve_retweets():
    """
    retrieve the first 100 retweets using Twitter API

    :return:
    """
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


def scrape_retweets():
    """
    collect retweets using twarc scraper
    :return: jsonl
    """
    # for tweet in t.retweets("525068387899031552"):
    for tweet in t.retweets("529740161144602624"):
        pp(tweet)
        print(tweet["text"])

# retrieve_retweets()
scrape_retweets()


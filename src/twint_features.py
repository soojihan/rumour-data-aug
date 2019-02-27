import twint
import glob
import pandas as pd
from pprint import pprint as pp
import os
import tweepy
import json
consumer_key =  "yTaqhLcxY2SJTcJSbe9CTlfWI"
consumer_secret = "au6Mj1lMGzlnDkkqcZrCoowzylJ0yxsAlmVLXRKDwpUkU7uXUu"
access_token = "803794891000205314-bs7wteyEIAxuETlcxkjOyvtrCphiKsD"
access_token_secret = "1BGjkLMRxDrNy4oqxqxAIEMLYQrBEZzcZAvvcClozizPX"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


c = twint.Config()

c.User_id = '1960878613'

# c.User_full = False
# c.Format = "Username: {username} | Bio: {bio} | Url: {url}"

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
            final_df = final_df.append(df)
    return final_df


# df = load_data('../data/pheme-training')
# df = load_data('../data/train')
df_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train')
print("df path ", df_path)
df = load_data(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train'))

pp(list(df))
df.drop_duplicates(subset=['tweet_id'], inplace=True)
print(len(df))
# print(len(set(df['tweet_id'].values)))
# print(len(set(df['user_id'].values)))

def retrieve_retweets():
    """
    retrieve the first 100 retweets

    :return:
    """
    for i, row in df.iterrows():
        user_id = row['user_id']
        tweet_id = row['tweet_id']
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/train_test/{}'.format(tweet_id))
        if os.path.exists(output_path):
            continue
        else:
            try:
                results = api.retweets(tweet_id)
                print("Tweet id: {}, num of retweets: {}".format(tweet_id, len(results)))
                # output_path = '../data/train_test/{}'.format(tweet_id)
                print(output_path)
                os.makedirs(output_path, exist_ok=True)
                retweets_json = []
                for obj in results:
                    pp(obj._json)
                    # retweets_json.append(obj._json)
                # with open(os.path.join(output_path, 'retweets.json'), 'w') as f:
                #     json.dump(retweets_json, f, indent=1)
                    raise SystemExit
            except tweepy.TweepError as e:
                print('Something went wrong, quitting...', e)



# def get_user_followees():


# retrieve_retweets()

import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup as bs
import time
import os
from pprint import pprint as pp
from glob import glob
import re
import pandas as pd
import pickle
import sys

pattern = '\d+'
re.compile(pattern)
def init_driver():
    # initiate the driver:
    # driver = webdriver.Chrome()
    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'contexts', 'chromedriver'))

    # set a default wait time for the browser [5 seconds here]:
    driver.wait = WebDriverWait(driver, 5)

    return driver


def close_driver(driver):
    driver.close()

    return

def login_twitter(driver, username, password):
    # open the web page in the browser:
    driver.get("https://twitter.com/login")

    # find the boxes for username and password
    username_field = driver.find_element_by_class_name("js-username-field")
    password_field = driver.find_element_by_class_name("js-password-field")

    # enter your username:
    username_field.send_keys(username)
    driver.implicitly_wait(1)

    # enter your password:
    password_field.send_keys(password)
    driver.implicitly_wait(1)

    # click the "Log In" button:
    driver.find_element_by_class_name("EdgeButtom--medium").click()

    return


class wait_for_more_than_n_elements_to_be_present(object):
    def __init__(self, locator, count):
        self.locator = locator
        self.count = count

    def __call__(self, driver):
        try:
            elements = EC._find_elements(driver, self.locator)
            return len(elements) > self.count
        except StaleElementReferenceException:
            return False


def search_twitter(driver, user_name, tweet_id):
    # wait until the search box has loaded:
    # box = driver.wait.until(EC.presence_of_element_located((By.NAME, "q")))

    driver.base_url = "https://twitter.com/" + user_name + "/status/" + tweet_id
    driver.get(driver.base_url)

    # initial wait for the search results to load
    wait = WebDriverWait(driver, 10)

    try:
        # wait until the first search result is found. Search results will be tweets, which are html list items and have the class='data-item-id':
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "li[data-item-id]")))
        while True:

            # extract all the tweets:
            tweets = driver.find_elements_by_css_selector("li[data-item-id]")

            # find number of visible tweets:
            number_of_tweets = len(tweets)

            # keep scrolling:
            driver.execute_script("arguments[0].scrollIntoView();", tweets[-1])

            try:
                # wait for more tweets to be visible:
                wait.until(wait_for_more_than_n_elements_to_be_present(
                    (By.CSS_SELECTOR, "li[data-item-id]"), number_of_tweets))

            except TimeoutException:
                # if no more are visible the "wait.until" call will timeout. Catch the exception and exit the while loop:
                break


        # extract the html for the whole lot:
        page_source = driver.page_source



    except TimeoutException:

        # if there are no search results then the "wait.until" call in the first "try" statement will never happen and it will time out. So we catch that exception and return no html.
        page_source = None

    return page_source


def extract_tweets(page_source):
    soup = bs(page_source, 'lxml')

    tweets = []
    reply_ids = []
    for li in soup.find_all("li", class_='js-stream-item'):

        # If our li doesn't have a tweet-id, we skip it as it's not going to be a tweet.
        if 'data-item-id' not in li.attrs:
            continue

        else:
            # tweet = {
            #     'tweet_id': li['data-item-id'],
            #     'text': None,
            #     'user_id': None,
            #     'user_screen_name': None,
            #     'user_name': None,
            #     'created_at': None,
            #     'retweets': 0,
            #     'likes': 0,
            #     'replies': 0
            # }
            #
            # # Tweet Text
            # text_p = li.find("p", class_="tweet-text")
            # if text_p is not None:
            #     tweet['text'] = text_p.get_text()
            #
            # # Tweet User ID, User Screen Name, User Name
            # user_details_div = li.find("div", class_="tweet")
            # if user_details_div is not None:
            #     tweet['user_id'] = user_details_div['data-user-id']
            #     tweet['user_screen_name'] = user_details_div['data-screen-name']
            #     tweet['user_name'] = user_details_div['data-name']
            #
            # # Tweet date
            # date_span = li.find("span", class_="_timestamp")
            # print(date_span)
            # if date_span is not None:
            #     tweet['created_at'] = float(date_span['data-time-ms'])
            #
            # # Tweet Retweets
            # retweet_span = li.select("span.ProfileTweet-action--retweet > span.ProfileTweet-actionCount")
            # if retweet_span is not None and len(retweet_span) > 0:
            #
            #     tweet['retweets'] = int(retweet_span[0]['data-tweet-stat-count'])
            #
            # # Tweet Likes
            # like_span = li.select("span.ProfileTweet-action--favorite > span.ProfileTweet-actionCount")
            # if like_span is not None and len(like_span) > 0:
            #     tweet['likes'] = int(like_span[0]['data-tweet-stat-count'])

            # Tweet Replies
            # reply_span = li.select("span.ProfileTweet-action--reply > span.ProfileTweet-actionCount > span.ProfileTweet-actionCountForAria" )
            reply_span = li.select("span.ProfileTweet-action--reply > span.ProfileTweet-actionCount")
            print(reply_span)
            if reply_span is not None and len(reply_span) > 0:
                reply_id = reply_span[0]['id']
                id = re.findall(pattern, reply_id)
                reply_ids.append(id[0])
                # tweet['replies'] = int(reply_span[0]['data-tweet-stat-count'])
                # tweet['replies'] = int(reply_span[0]['id'])

            # tweets.append(tweet)

    return tweets, reply_ids

def load_abs_path(data_path: str)-> str:
    """
    read actual data path from either symlink or a absolute path

    :param data_path: either a directory path or a file path
    :return:
    """
    if not os.path.exists(data_path) or os.path.islink(data_path):
        if sys.platform == "win32":
            return readlink_on_windows(data_path)
        else:
            return os.readlink(data_path)

    return data_path

if __name__ == "__main__":
    # start a driver for a web browser:


    # files = glob(os.path.join(os.getcwd(), '..', 'data_augmentation', 'data', 'augmented_data_annotation', 'boston', 'boston-7500.csv'))
    # files = glob(os.path.join(os.getcwd(), '..', 'data_augmentation', 'data', 'file-embed-output', 'boston', 'results_p8502<0.3', 'boston-8502.csv'))
    files = glob(os.path.join(os.getcwd(), '..', 'data_augmentation', 'data', 'file-embed-output', 'boston', 'results_p9003<0.3', 'boston-9003.csv'))
    pp(files)
    #
    outpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'data_augmentation', 'data', 'augmented_data', 'boston-9003-part2'))
    # outpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'data_augmentation', 'data', 'augmented_data', 'boston-8500'))
    print(outpath)



    os.makedirs(outpath, exist_ok=True)
    # driver = init_driver()
    # # log in to twitter (replace username/password with your own):
    # username = 'mujisuji'
    # password = 'london2paris'
    # login_twitter(driver, username, password)
    driver = init_driver()
    # log in to twitter (replace username/password with your own):
    username = 'mujisuji'
    password = 'london2paris'
    login_twitter(driver, username, password)

    for f in files:
        df = pd.read_csv(f)
        print(list(df))
        print(len(df))
        very_start = time.time()
        # df = df[3787:]
        for i, row in df.iterrows():
            start = time.time()
            user_name = row['screen_name'] # source tweet username
            tweet_id = str(int(row['id'])) # source tweet id
            print(i, tweet_id)

            label = row['label']
            if os.path.exists(os.path.join(outpath, 'non-rumours', '{}'.format(tweet_id))) or \
                    os.path.exists(os.path.join(outpath, 'rumours', '{}'.format(tweet_id))):
                print("Already exists.. pass ")
                continue

            # driver.base_url = "https://twitter.com/" + user_name + "/status/" + tweet_id
            # driver.get(driver.base_url)
            # page_source = driver.page_source
            page_source = search_twitter(driver, user_name, tweet_id)
            if page_source is None:
                print("empty replies")
                print("")
                continue
            else:
                _, reply_ids = extract_tweets(page_source)
                print("Replies exist " ,tweet_id, len(reply_ids))
                print("")
                # if (label == 0) and (len(reply_ids)>0):
                #     parent_path = os.path.join(outpath, 'non-rumours', '{}'.format(tweet_id))
                #     os.makedirs(parent_path, exist_ok=True)
                #
                # elif (label==1) and (len(reply_ids)>0):
                #     parent_path = os.path.join(outpath, 'rumours', '{}'.format(tweet_id))
                #     os.makedirs(parent_path, exist_ok=True)
                # else:
                #     continue

                # with open(os.path.join(parent_path, 'reactions_ids.pickle'), 'wb') as f:
                #     pickle.dump(reply_ids, f)
                #     f.close()
                #
                # with open(os.path.join(parent_path, 'reactions_ids.pickle'), 'rb') as f:
                #     x = pickle.load(f)
                #     # print(len(x))
                end = time.time()
                print(end-start)
                if i%1000==0:
                    print("Tweet: {}/{} , So far......... {} ". format(i, len(df), end-very_start ))
                # if i == 3:
                #     break
    close_driver(driver)
    driver.quit()

#
# driver = init_driver()
#     # log in to twitter (replace username/password with your own):
# username = 'mujisuji'
# password = 'london2paris'
# login_twitter(driver, username, password)
#
# USERNAME = "Cristiano"
# TWEETID = "1107631650219921408"
# # driver.base_url = "https://twitter.com/" + USERNAME + "/status/" + TWEETID
# # driver.get(driver.base_url)
# # page_source = driver.page_source
# # tweets, reply_ids = extract_tweets(page_source)
# # pp(reply_ids)
#
# page_source = search_twitter(driver, USERNAME, TWEETID)
# # if page_source is None:
# #     print("empty replies")
# #     print("")
# #     continue
# # else:
# _, reply_ids = extract_tweets(page_source)
# pp(reply_ids)
# close_driver(driver)
# #
#
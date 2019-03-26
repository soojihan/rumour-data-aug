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

"""
Scrape reply ids of source tweets
"""
pattern = '\d+'
re.compile(pattern)

def init_driver():
    # initiate the driver:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-setuid-sandbox')

    driver = webdriver.Chrome(os.path.join(os.getcwd(), 'chromedriver'), options=chrome_options)

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

            # Tweet Replies
            reply_span = li.select("span.ProfileTweet-action--reply > span.ProfileTweet-actionCount > span.ProfileTweet-actionCountForAria" )
            # reply_span = li.select("span.ProfileTweet-action--reply > span.ProfileTweet-actionCount")
            print(reply_span)
            if reply_span is not None and len(reply_span) > 0:
                reply_id = reply_span[0]['id']
                id = re.findall(pattern, reply_id)
                reply_ids.append(id[0])

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


    files = glob(os.path.join('..', '..', 'data_augmentation', 'data', 'file-embed-output', 'boston', 'results_p9003<0.3', 'boston-9003.csv'))
    pp(files)
    #
    outpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'data_augmentation', 'data', 'augmented_data', 'boston-9003-part2'))
    print(outpath)



    os.makedirs(outpath, exist_ok=True)
    driver = init_driver()

    # log in to twitter (replace username/password with your own):
    username = ''
    password = ''
    login_twitter(driver, username, password)

    for f in files:
        df = pd.read_csv(f)
        print(list(df))
        print(len(df))
        very_start = time.time()
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
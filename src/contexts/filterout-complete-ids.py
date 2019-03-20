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


files = glob(os.path.join(os.getcwd(), '..', '..', 'data_augmentation', 'data', 'augmented_data_annotation', 'boston', 'boston-7500*.csv'))
pp(files)

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

outpath = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data_augmentation', 'data', 'augmented_data', 'boston'))
print(outpath)
outpath = load_abs_path(outpath)
print(outpath)

# for f in files:
#     df = pd.read_csv(f)
#     for i, row in df.iterrows():
#         user_name = row['screen_name']  # source tweet username
#         tweet_id = str(int(row['id']))  # source tweet id
#         label = row['label']
#         if os.path.exists(os.path.join(outpath, 'non-rumours', '{}'.format(tweet_id))) or \
#                 os.path.exists(os.path.join(outpath, 'rumours', '{}'.format(tweet_id))):
#             print("Already exists.. pass ")
#             with open(os.path.join(outpath, 'complete-ids.txt'), 'a') as f:
#                 f.write(tweet_id)
#                 f.write('\n')
#                 f.close()

with open(os.path.join(outpath, 'complete-ids.txt'), 'r') as f:
    x = f.read().splitlines()
    pp(x)
    f.close()
print(type(x[0]))
df_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       '..', 'data_augmentation/data/augmented_data_annotation/boston'))

## text files
outfile = os.path.join(df_path, 'boston-7500-source-ids.txt')
with open(outfile, 'r') as f:
    y = f.read().splitlines()
    f.close()
outfile = os.path.join(df_path, 'boston-7500-source-ids-filtered.txt')
for i in y:
    if i not in x:
        with open(outfile, 'a') as f:
            f.write(str(int(i)))
            f.write('\n')
        f.close()

## csv
outfile = os.path.join(df_path, 'boston-7500.csv')
df = pd.read_csv(outfile)
print(len(df))
new_df = pd.DataFrame(columns=list(df))
for i, row in df.iterrows():
    if str(int(row['id'])) not in x:
        new_df.loc[len(new_df)]= row
print(len(new_df))
new_df.to_csv(os.path.join(df_path, 'boston-7500-filtered.csv'))
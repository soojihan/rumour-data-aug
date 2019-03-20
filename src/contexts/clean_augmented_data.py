import os
import pandas as pd
import pickle
import numpy as np
import json
from glob import glob
from pprint import pprint as pp
import shutil
"""
Move source tweets and retweets to replies folder
"""
pd.set_option('display.expand_frame_repr', False)

def load_abspath(x):
    return os.path.abspath(x)


def merge_all_types():
    """
    1. Move source tweets and retweets to reply folder
    :return:
    """
    reply_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data/augmented_data_temp/boston/replies/boston-9003-re'))
    source_retweet_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data/augmented_data_temp/boston/sourcetweets_retweets/hydrates-9003'))
    assert os.path.exists(reply_path)
    assert os.path.exists(source_retweet_path)

    source_ids = load_abspath(os.path.join('..', '..', 'data_augmentation/data/file-embed-output/boston/results_p9003<0.3/boston-9003.csv'))
    source_ids = pd.read_csv(source_ids)


    for i, row in source_ids.iterrows():
        source_id = str(int(row['id']))
        print("Source id ", source_id)
        if not glob(load_abspath(os.path.join(reply_path, '*', '{}'.format(source_id)))):
            print("source tweet does not have any reply")
            continue
        else:
            ## Remove source tweet directories where no reply exists
            path_to_check = glob(load_abspath(os.path.join(reply_path, '*', '{}/reactions'.format(source_id))))
            if not path_to_check:
                print(glob(load_abspath(os.path.join(reply_path, '*', '{}'.format(source_id))))[0])
                print(path_to_check)
                shutil.rmtree(glob(load_abspath(os.path.join(reply_path, '*', '{}'.format(source_id))))[0])

            else:
                print("Move retweets and source tweets ")
                # parent_path = os.path.join(source_retweet_path, '{}'.format(source_id))
                dest_path = glob(load_abspath(os.path.join(reply_path, '*', '{}'.format(source_id))))[0]
                # if os.path.exists(parent_path):
                #     subdirs = os.listdir(parent_path)
                #     subdirs = [x for x in subdirs if not x.startswith(".")]
                #     pp(subdirs)
                #     retweet_path = os.path.join(parent_path, subdirs[0])
                #     source_path = os.path.join(parent_path, subdirs[1])
                #     print(retweet_path)
                #     print(source_path)
                #     print(dest_path)
                #     if not os.path.exists(os.path.join(dest_path, subdirs[0])):
                #         shutil.copytree(retweet_path, os.path.join(dest_path, subdirs[0]))
                #     if not os.path.exists(os.path.join(dest_path, subdirs[1])):
                #         shutil.copytree(source_path, os.path.join(dest_path, subdirs[1]))

                hydrator_data = os.path.join('..', '..',
                                             'data_augmentation/data/file-embed-output/boston/results_p9003<0.3/hydrator_results')
                df = pd.read_csv(hydrator_data)

                source_subset = df[df.id == int(source_id)]
                print(source_subset)
                source_json = {}
                if not source_subset.empty:
                    source_json['text'] = source_subset.iloc[0]['text']
                    source_json['created_at'] = source_subset.iloc[0]['created_at']
                    source_json['id'] = str(source_subset.iloc[0]['id'])
                    pp(source_json)
                    outpath = os.path.join(dest_path, 'source-tweets')
                    os.makedirs(outpath, exist_ok=True)
                    outpath = os.path.join(outpath, '{}.json'.format(str(source_json['id'])))
                    with open(outpath, 'w') as f:
                        json.dump(source_json, f)
                        # raise SystemExit
def remove_dirs():
    """
    2. Remove dirs that do not contain source tweets and retweets
    :return:
    """
    reply_path = load_abspath(
        os.path.join('..', '..', 'data_augmentation/data/augmented_data_temp/boston/replies/boston-9003-re'))
    source_ids = load_abspath(
        os.path.join('..', '..', 'data_augmentation/data/file-embed-output/boston/results_p9003<0.3/boston-9003.csv'))
    source_ids = pd.read_csv(source_ids)

    for i, row in source_ids.iterrows():
        source_id = str(int(row['id']))
        print("Source id ", source_id)
        x = glob(load_abspath(os.path.join(reply_path, '*', '{}'.format(source_id))))
        print(x)
        if x:
            subdirs = os.listdir(x[0])
            print(subdirs)
            ## Check if this dir has source tweets
            if 'source-tweets' not in subdirs:
                ## If not, remove the dir
                shutil.rmtree(glob(load_abspath(os.path.join(reply_path, '*', '{}'.format(source_id))))[0])

def clean_source_ids():
    """
    3. Generate clean files containing source tweet ids and labels
    :return:
    """
    source_tweets = os.path.join("..", '..', 'data_augmentation/data/augmented_data_annotation/boston/results_p9003<0.3')
    originaldf = pd.read_csv(os.path.join(source_tweets,'boston-9003.csv' ))
    dpath = load_abspath(
        os.path.join('..', '..', 'data_augmentation/data/augmented_data/boston'))
    subdirs = glob(os.path.join(dpath, '*', '*'))
    source_ids = []
    outdf = pd.DataFrame(columns=list(originaldf))

    for sdir in subdirs:
        source_id = os.path.basename(sdir)
        print(source_id)
        source_ids.append(source_id)
        subset = originaldf[originaldf.id == int(source_id)]
        print(subset)
        outdf = pd.concat((outdf, subset))
    print(outdf)
    outdf.drop(['Unnamed: 0'], inplace=True, axis=1)
    outdf.reset_index(drop=True, inplace=True)

    # outdf['tweet_id'] = source_ids
    outdf.to_csv(os.path.join(source_tweets, 'clean-sourceids.csv'))

# merge_all_types()
# remove_dirs()
clean_source_ids()

## Hydrator version


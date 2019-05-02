import os
import pandas as pd
import pickle
import numpy as np
import json
from glob import glob
from pprint import pprint as pp
import shutil
import jsonlines
import random

"""
(Re-)arrange source tweets,retweets, replies to match the PHEME's format 
"""
pd.set_option('display.expand_frame_repr', False)

def load_abspath(x):
    return os.path.abspath(x)


def arrange_source_tweets(event, data_path, save_path):
    """
    Save source tweet json object in /event/rumour_type/source_id/source-tweets/source_id.json
    All source tweets stored in JSON Lines format are downloaded using ttps://github.com/DocNow/hydrator
    """

    ## Save only augmented and sampled ids (the output of /src/data_augmentation.py)
    print("path to downloaded candidate tweets ", data_path)
    print("path to augmented data ", save_path)
    val_source_ids = {}
    for root, dirs, files in os.walk(save_path):
        valid_source_id = os.path.basename(root)
        print(valid_source_id)
        val_source_ids[valid_source_id] = root

    with jsonlines.open(os.path.join(data_path, '{}.jsonl'.format(event))) as reader:
        for obj in reader:
            source_id = obj['id_str']
            if 'text' in obj:
                text = obj['text']
            elif 'full_text' in obj:
                text = obj['full_text']
            else:
                raise ValueError
            # Check if this tweet is our sample
            if source_id in val_source_ids:
                # print(source_id)
                outpath = os.path.join(val_source_ids[source_id], 'source-tweets')
                os.makedirs(outpath,exist_ok=True)
                with open(os.path.join(outpath, '{}.json'.format(source_id)), 'w') as f:
                    json.dump(obj, f)

def arrange_replies(reply_path):
    """
    Save reply json object in /event/rumour_type/source_id/reactions/reply_id.json
    All replies stored in JSON Lines format are downloaded using ttps://github.com/DocNow/hydrator
    :param reply_path: path to downloaded replies of an event
    """
    with open(os.path.join(reply_path, 'reply_dict.pickle'), 'rb') as f: # Load reply dictionary (refer to 'src/reply_scraper.py')
        reply_dict = pickle.load(f)
        f.close()
    subdirs = [x[0] for x in os.walk(reply_path)]

    with jsonlines.open(os.path.join(reply_path, 'all_replies.jsonl')) as reader: # Load all downloaded replies for an event
        for obj in reader:
            reply_id = obj['id_str']
            print("")
            print("Reply id ", reply_id)

            for source_id, replies in reply_dict.items():
                if reply_id in replies:
                    print("Source id ", source_id, "Reply id ", reply_id)
                    save_path = glob(os.path.join(reply_path, '*', '{}'.format(source_id)))[0]
                    save_path = os.path.join(save_path, 'reactions')
                    print("Path to save reply objects ", save_path)
                    os.makedirs(save_path, exist_ok=True)
                    with open(os.path.join(save_path, '{}.json'.format(reply_id)), 'w') as f:
                        json.dump(obj, f)


def remove_empty_dirs(data_path: str):
    """
    Remove directories that do not contain certain folder(s) such as 'reactions' and 'source tweets'
    """
    subdirs = glob(load_abspath(os.path.join(data_path, 'non-rumours', '*')))
    # subdirs = glob(load_abspath(os.path.join(data_path, 'rumours', '*')))
    # subdirs = glob(load_abspath(os.path.join(data_path, '*', '*')))
    count = 0
    for dir in subdirs:
        print(dir)
        subdir = os.listdir(dir)
        print(subdir)
        source_id = os.path.basename(dir)
        # if 'source-tweets' not in subdir:
        if ('reactions' not in subdir):
            count +=1
            # print(glob(load_abspath(os.path.join(data_path, 'non-rumours', '{}'.format(source_id))))[0])
            # shutil.rmtree(glob(load_abspath(os.path.join(data_path, 'non-rumours', '{}'.format(source_id))))[0])
    print(count)


def balance_data(data_path: str):
    """
    Remove randomly selected non-rumours to balance class distributions
    """
    ## All non-rumour source tweet dirs
    subdirs = glob(load_abspath(os.path.join(data_path, 'non-rumours', '*')))

    ## Get number of rumours to decide sample size for non-rumours
    rumour_subdirs = glob(load_abspath(os.path.join(data_path, 'rumours', '*')))
    num_rumours = len(rumour_subdirs)
    sample_size = 2*num_rumours
    print(sample_size)
    count = 0
    print(len(subdirs))
    large_reactions = []
    # selected_subdirs = []
    for dir in subdirs:
        subdir = os.listdir(dir)
        source_id = os.path.basename(dir)
        # if dir not in selected_subdirs:
        if (len(glob(os.path.join(dir, 'reactions', '*')))>10):
            count +=1
            large_reactions.append(dir)

    print("Number of subdirs with large reactions ", len(large_reactions))
    selected_subdirs = random.sample(set(subdirs)-set(large_reactions), sample_size-len(large_reactions))
    print("Number of randomly selected subdirs ", len(selected_subdirs))
    final_sampled_sets = selected_subdirs + large_reactions
    print("Number of final non-rumour samples", len(final_sampled_sets))
    assert len(final_sampled_sets) == sample_size

    ## Remove non-rumour dirs which are not included in the final sample
    count = 0
    for dir in subdirs:
        if dir not in final_sampled_sets:
            count+=1
            shutil.rmtree(glob(load_abspath(os.path.join(data_path, 'non-rumours', '{}'.format(source_id))))[0])
    print("Number of deleted non-rumor source tweets: ", count)

def generate_final_metadata(final_aug_path):
    """
    Generate files containing metadata of augmented data
    Needed for 'Multitask4Veracity' ->/Multitask4Veracity/detection_only/preprocessing.py preprocessing_context
    :return:
    """
    subdirs = glob(os.path.join(final_aug_path, '*', '*')) # pheme
    # subdirs = glob(os.path.join(final_aug_path, 'saved_tweets', '*', '*')) # twitter1516

    pp(subdirs)
    final_df = pd.DataFrame(columns=['tweet_id', 'text', 'created_at', 'label', 'user_id'], dtype=str)
    for i, sdir in enumerate(subdirs):
        rumour_type= os.path.basename(os.path.dirname(sdir))
        source_id = os.path.basename(sdir)
        print(source_id)
        assert type(source_id) == str
        label = 1 if rumour_type == 'rumours' else 0
        source_tweet_json = os.path.join(sdir, 'source-tweets', '{}.json'.format(source_id))
        assert os.path.exists(source_tweet_json)
        with open(source_tweet_json, 'r') as f:
            source_obj= json.load(f)
            f.close()
        if 'full_text' in source_obj:
            source_text = source_obj['full_text']
        elif 'text' in source_obj:
            source_text =source_obj['text']
        else:
            raise ValueError
        created_at = source_obj['created_at']
        user_id = source_obj['user']['id_str']
        final_df.loc[len(final_df)] = [source_id, source_text, created_at, label, user_id]

    print(len(final_df))
    print(final_df)
    assert len(final_df) == len(subdirs)
    final_df.to_csv(os.path.join(final_aug_path, 'aug_metadata.csv'))
    with open(os.path.join(final_aug_path, 'aug_metadata.pickle'), 'wb') as f:
        pickle.dump(final_df, f)
        f.close()



def main():
    """
    -------------------------------------------------------------------------------
    event: event name
    data_path: path to candidates downloaded using Hydrator (in JSON Lines format)
    save_path: path to save restructured files (e.g., source tweets)
    --------------------------------------------------------------------------------
    """
    event = 'ottawashooting'

    ## Path to augmented source tweets
    data_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/downloaded_data/hydrator'))

    save_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/augmented_data/{}'.format(event)))

    # arrange_source_tweets(event, data_path, save_path)
    # arrange_replies( reply_path=save_path)
    # balance_data(save_path)

    ## Augmented data + the original pheme
    # final_aug_path = load_abspath(os.path.join('..', '..', 'data_augmentation/data_hydrator/augmented_data/{}'.format(event)))
    generate_final_metadata(final_aug_path)

main()
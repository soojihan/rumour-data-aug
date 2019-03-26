import pandas as pd
import os
from glob import glob

def data_augmentation(event: str = 'sydneysiege'):
    """
    Deduplicate the final output (after filtering out by applying a threshold)
    Balance pos and eng examples using the threshold fine tuned usnig the SemEval
                                     threshold   precision
    # subset = df[df['sim_score'] >= 0.652584] # 6088
    # subset = df[df['sim_score'] >= 0.691062] # 7000
    # subset = df[df['sim_score'] >= 0.708341] # 7500
    # subset = df[df['sim_score'] >= 0.760198] # 8502
    # subset = df[df['sim_score'] >= 0.801806] # 9003
    # subset = df[df['sim_score'] >= 0.849272] # 9506

    :return:
    """
    files = glob(os.path.join(os.path.dirname(__file__), '..', 'data_augmentation/data/file-embed-output/{}/scores/ref*.csv'.format(event)))
    pos_ids = set()
    neg_ids = set()

    pos_threshold = 0.801806
    precision = 9003
    # neg_threshold = 0.33
    neg_threshold = 0.3
    for f in files: # iter each reference result
        df = pd.read_csv(f)

        ## Select positive examples using a fine-tuned threshol
        pos_subset = df[df['sim_score'] >= pos_threshold]
        sub_pos_ids = list(pos_subset['id'].values)

        ## Sample negative examples
        # TODO: Improve a method for sampling negative examples
        neg_subset = df[ df['sim_score'] < neg_threshold]
        # false_neg_subset = df[df['sim_score']>=neg_threshold]
        # false_neg_subset = df[df['sim_score'] <= neg_threshold]

        sub_neg_ids = list(neg_subset['id'].values)
        # false_neg_ids = list(false_neg_subset['id'].values)
        pos_ids.update(sub_pos_ids)
        neg_ids.update(sub_neg_ids)
        print("Updated negative indices ", len(neg_ids))
        # neg_ids -= set(false_neg_ids)
        # print("After removing false indices ", len(neg_ids))
        print("")
    print("*"*10)

    print("Number of positive examples ", len(pos_ids))
    print("Number of negative examples ", len(neg_ids))
    print("")

    num_pos = len(pos_ids)
    infile = '/Users/suzie/Desktop/PhD/Projects/data-aug/data_augmentation/data/file-embed-output/{}/'.format(event)
    full_df = pd.read_csv(os.path.join(infile, 'dropped_candidates.csv'))
    total_indices = list(full_df.index)
    print("Total number of candidates ", len(total_indices))

    random_neg_indices = random.sample(neg_ids, num_pos*3)
    print(random_neg_indices)
    
    ## Find intersection -> if a tweet appears in both negative and positive samples, remove it from negative set
    intersection = pos_ids.intersection(random_neg_indices)
    print(intersection)
    print("before removing intersection ", len(intersection), len(random_neg_indices))
    random_neg_indices = set(random_neg_indices)-intersection
    print("after removing intersection ", len(random_neg_indices))


    ## Generate final input df
    full_df.drop(['Unnamed: 0'], inplace=True, axis=1)
    pos_subset = full_df.loc[pos_ids]
    pos_subset['label'] = np.ones(num_pos, dtype=int)
    neg_subset = full_df.loc[random_neg_indices]
    # neg_subset = neg_subset[neg_subset['retweet_count']<100] # filter out negative subset
    # pp(list(neg_subset['text'].values))
    neg_subset['label'] = np.zeros(len(neg_subset), dtype=int)
    result = pd.concat([pos_subset, neg_subset])
    print(len(result))
    save_path = os.path.join(infile, 'results_p{}<0.3'.format(precision))
    os.makedirs(save_path, exist_ok=True)
    result.to_csv(os.path.join(save_path, '{}-{}.csv'.format(event, precision)))

# TODO: a tweet can be a rumour for a reference but a non-rumour for another reference --> consier it as a rumour
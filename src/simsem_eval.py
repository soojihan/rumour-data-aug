import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def eval(result_path):
    """
    Find the best threshold which achieves the maximum F-measure
    :return:
    """
    with open(result_path, 'r') as f:
        df = pd.read_csv(f)
        print(df.head())
    print(len(df))
    df.rename(columns={'label': 'goldlabel'}, inplace=True)
    df.rename(columns={'sim_score': 'sim_scores'}, inplace=True)
    # df.rename(columns={'binary_label': 'goldlabel'}, inplace=True)

    df[['goldlabel']] = df[['goldlabel']].astype(int)

    # df.sort_values(by='sim_scores', ascending=True, inplace=True)
    threshold_candidates = list(set(df['sim_scores'].values))
    # threshold_candidates = [x for x in threshold_candidates if x>=0.5]
    threshold_candidates.sort(reverse=False)
    print(threshold_candidates)
    # threshold_candidates.sort(reverse=True)
    print(threshold_candidates)
    print(len(threshold_candidates))
    print(df.head())
    max_F = 0
    max_P = 0
    max_R = 0
    max_SP = 0
    optimum_threshold = 0
    ps = []
    rs =[]
    threshs=[]
    for threshold in threshold_candidates:
        df.loc[df[df.sim_scores >= threshold].index, 'syslabel'] = 1
        df.loc[df[df.sim_scores < threshold].index, 'syslabel'] = 0
        syslabels = df['syslabel'].values
        goldlabels = df['goldlabel'].values
        F, P, R, SP = get_eval_metrics(syslabels, goldlabels)
        print(F,P,R)
        # if max_F < F:
        if max_P <= P:
        # if max_SP < SP:
            max_F = F
            max_P = P
            max_R = R
            max_SP = SP
            optimum_threshold = threshold
            ps.append(P)
            rs.append(R)
            threshs.append(optimum_threshold)
            # print("max F-measure: {:0.4f}, P: {:0.4f}, R: {:0.4f}, threshold: {:0.6f}".format(max_F, max_P, max_R, optimum_threshold))
            print("max P: {:0.4f}, F: {:0.4f}, R: {:0.4f}, SP: {:0.4f}, threshold: {:0.6f}".format(max_P, max_F, max_R, max_SP, optimum_threshold))
        # if threshold > 0.33:
        #     break
    return ps, rs, threshs

def get_eval_metrics(syslabels, goldlabels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for (i, syslabel) in enumerate(syslabels):

        # if syslabel == True and goldlabels[i] == True:
        if syslabel == 1 and goldlabels[i] == 1:
            tp += 1
        elif syslabel == 1 and goldlabels[i] == 0:
            fp += 1
        elif syslabel == 0 and goldlabels[i] == 0:
            tn += 1
        elif syslabel == 0 and goldlabels[i] == 1:
            fn += 1

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    SP = tn / (tn+fp) # Specifity
    # print("tp {}, fp {}, tn {}, fn {}".format(tp, fp, tn, fn))
    testsize = str(tp + fn + fp + tn)
    return F, P, R, SP

def plot_pr_curve():
    result_path = os.path.join('..',  'data/semeval2015/results/elmo_credbank/{}.csv'.format('elmo_merged_55b'))
    print(os.path.abspath(result_path))
    ps, rs, threshs = eval(result_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(threshs, ps, 'k--', linewidth=3, label='Precision')
    ax.plot(threshs, rs, 'k', linewidth=3, label='Recall')
    # plt.xlabel('Threshold')
    ax.set_xlabel('Threshold', fontsize=36)
    plt.xticks(fontsize=33)
    plt.yticks(fontsize=36)
    plt.legend(loc='lower left', fontsize=34)
    plt.tight_layout()
    plt.show()

# plot_pr_curve()

# elmo_semantic_similarity()
# eval_results()
# prepare_elmo_embeddings()

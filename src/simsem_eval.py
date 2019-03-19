import pandas as pd
import os

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
    threshold_candidates = [x for x in threshold_candidates if x>=0.5]
    threshold_candidates.sort(reverse=False)
    print(len(threshold_candidates))
    print(df.head())
    max_F = 0
    max_P = 0
    max_R = 0
    optimum_threshold = 0
    for threshold in threshold_candidates:
        # print("Threshold ", threshold)
        df.loc[df[df.sim_scores >= threshold].index, 'syslabel'] = 1
        df.loc[df[df.sim_scores < threshold].index, 'syslabel'] = 0
        syslabels = df['syslabel'].values
        goldlabels = df['goldlabel'].values
        F, P, R = get_eval_metrics(syslabels, goldlabels)
        # if max_F < F:
        if max_P < P:
            max_F = F
            max_P = P
            max_R = R
            optimum_threshold = threshold
            # print("max F-measure: {:0.4f}, P: {:0.4f}, R: {:0.4f}, threshold: {:0.6f}".format(max_F, max_P, max_R, optimum_threshold))
            print("max P: {:0.4f}, F: {:0.4f}, R: {:0.4f}, threshold: {:0.6f}".format(max_P, max_F, max_R, optimum_threshold))


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

    testsize = str(tp + fn + fp + tn)
    return F, P, R
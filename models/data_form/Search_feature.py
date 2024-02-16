import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import pandas as pd
import numpy as np 
import os
from functools import partial

from codes.models.data_form.DataForm import DataTransfo_1SNP, PatientList
import matplotlib.pyplot as plt
from codes.models.Naive.Naive_utils import get_pred_naive, get_pred_sentence, get_risk_pheno

method = 'Abby'
path_save_df = f'{path}codes/Data_Files/phewas/df_scores_SNPS_{method}.csv'



def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exp_x / exp_x.sum(axis=0)
### search for the SNPS 
dir_chrs = f'{path}codes/Data_Files/Training/SNPS'
phewas_cat_file = f'{path}codes/Data_Files/phewas/phewas-catalog.csv'


list_files = os.listdir(dir_chrs)
list_chrs = []
for chr in list_files:
    try:
        list_chrs.append(int(chr))
    except:
        pass


def get_score_SNP(chr, SNP,method, data_base=None):
    CHR = chr
    SNP = SNP
    if data_base==None:
        dataT = DataTransfo_1SNP(
            method=method,
            SNP=SNP,
            CHR=CHR,
            load_data=True,
            save_data=False
        )
        data_base, labels_base, indices_env, name_envs = dataT.get_tree_data()
    data, labels = DataTransfo_1SNP.equalize_label(data_base, labels_base)
    nb_phenos = data.shape[1]
    phenos = np.arange(nb_phenos)
    get_risk_pheno_par = partial(get_risk_pheno, data, labels)
    get_pred_naive_par = partial(get_pred_naive, data, labels)
    frequencies = data.sum(axis=0) / len(data)

    odds_ratios = np.array(list(map(get_risk_pheno_par, phenos)))
    pred_naive = np.array(list(map(get_pred_naive_par, phenos)))
    probas_pred_naive = pred_naive[:, 0]
    labels_pred_naive = pred_naive[:, 1]
    mask = (1 - np.isnan(probas_pred_naive)).astype(bool)
    data_masked = data[:, mask]
    probas_pred_naive_masked = probas_pred_naive[mask]
    labels_pred_naive_masked = labels_pred_naive[mask]
    odds_ratios_masked = odds_ratios[mask]
    frequencies_masked = frequencies[mask]

    weights_frequencies = softmax((probas_pred_naive_masked - 0.5)**2)
    frequencies_score = np.dot(frequencies_masked, weights_frequencies)
    probas_score = np.var(probas_pred_naive_masked)
    odds_ratios_score = np.var(odds_ratios_masked)

    #score of equalize
    imbalance_score = len(data) / len(data_base)
    return frequencies_score, probas_score, odds_ratios_score, imbalance_score

columns_score = ['CHR', 'SNP', 'frequencies', 'probas', 'odds_ratios', 'imbalance']
df_score_SNPS = pd.DataFrame(columns=columns_score)
for chr in list_chrs:
    dir_SNPS = f'{dir_chrs}/{str(chr)}'
    list_SNPS = os.listdir(dir_SNPS)
    for SNP in list_SNPS:
        print(chr, SNP)
        frequencies_score, probas_score, odds_ratios_score, imbalance_score = get_score_SNP(chr, SNP, method)
        df_scores = pd.DataFrame(columns = columns_score, data=[[chr, SNP, frequencies_score, probas_score, odds_ratios_score, imbalance_score]])
        df_score_SNPS = pd.concat([df_score_SNPS, df_scores], axis=0, ignore_index=True)
        df_score_SNPS.to_csv(path_save_df)
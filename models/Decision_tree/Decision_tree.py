print('debut test', flush=True)
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import os
import numpy as np
import pandas as pd
import time
import torch
import pickle
import shap
import tensorboard

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from functools import partial
import shutil
from tqdm.auto import tqdm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from codes.models.data_form.DataForm import DataTransfo_1SNP
from codes.models.metrics import calculate_roc_auc

import featurewiz as gwiz

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve





### framework constants:
model_type = 'decision_tree'
model_version = 'gradient_boosting'
test_name = '1_test_train_transfo_V1'
pheno_method = 'Abby' # Paul, Abby
tryout = True # True if we are ding a tryout, False otherwise 
### data constants:
### data constants:
CHR = 6
SNP = 'rs6903608'
pheno_method = 'Paul' # Paul, Abby
ld = 'no'
rollup_depth = 4
binary_classes = True #nb of classes related to an SNP (here 0 or 1)
vocab_size = None # to be defined with data
padding_token = 0
prop_train_test = 0.8
load_data = False
save_data = True
remove_none = True
decorelate = False
equalize_label = True
threshold_corr = 0.9
threshold_rare = 50
remove_rare = 'all' # None, 'all', 'one_class'
compute_features = True
padding = False
list_env_features = []
list_pheno_ids = None #list(np.load(f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/phewas/list_associations_snps/{SNP}_paul.npy'))

equalized = True
interest = False
keep = True
scaled = True
### data format

batch_size = 20
data_share = 1

##### model constants


##### training constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def decision_tree(SNP,
                CHR,
                method='Paul',
                padding=True,  
                pad_token=0, 
                load_data=True, 
                save_data=True, 
                compute_features=True,
                data_share=1,
                prop_train_test=0.8,
                remove_none=True,
                rollup_depth=4,
                binary_classes=True,
                ld='no',
                equalized=True,
                interest=False,
                keep=True,
                scaled=True,
                ):
    ##### training constants
    device = 'cpu'
    ##### model constants
    dataT = DataTransfo_1SNP(SNP=SNP,
                            CHR=CHR,
                            method=pheno_method,
                            padding=padding,  
                            pad_token=padding_token, 
                            load_data=load_data, 
                            save_data=save_data, 
                            compute_features=compute_features,
                            data_share=data_share,
                            prop_train_test=prop_train_test,
                            remove_none=remove_none,
                            rollup_depth=rollup_depth,
                            binary_classes=binary_classes,
                            ld=ld)


    # Définir les paramètres du modèle
    params = {'tree_method': 'gpu_hist', 'device': device, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}

   
    data, labels_patients, indices_env, name_envs = dataT.get_tree_data(with_env=False, load_possible=True, only_relevant=False)

    if interest:
        data_use, labels_use = data[:nb_patients_interest, :-1], labels_patients[:nb_patients_interest]
    else:
        data_use, labels_use = data, labels_patients

    if equalized:
        pheno, labels = DataTransfo_1SNP.equalize_label(data=data_use, labels = labels_use)
    else:
        pheno, labels = data_use, labels_use

    diseases_patients_train, diseases_patients_test, label_patients_train, label_patients_test = train_test_split(pheno, labels, test_size = 1-prop_train_test, random_state=42)

    class_weight = {0: np.sum(label_patients_train == 1) / np.sum(label_patients_train == 0), 1: 1.0}

    frequencies = np.sum(diseases_patients_train, axis=0)
    indices_keep = frequencies>100
    diseases_patients_train_keep = diseases_patients_train[:,indices_keep]
    diseases_patients_test_keep = diseases_patients_test[:, indices_keep]

    if keep:
        diseases_patients_train_model = diseases_patients_train_keep
        diseases_patients_test_model = diseases_patients_test_keep
    else:
        diseases_patients_train_model = diseases_patients_train
        diseases_patients_test_model = diseases_patients_test

        
    diseases_patients_train_model_unscaled = diseases_patients_train_model
    diseases_patients_test_model_unscaled = diseases_patients_test_model

    if scaled:

        scaler = StandardScaler()
        diseases_patients_train_model= scaler.fit_transform(diseases_patients_train_model)
        diseases_patients_test_model = scaler.fit_transform(diseases_patients_test_model)


    print(diseases_patients_train_model.shape)

    model = HistGradientBoostingClassifier(class_weight=class_weight)


    # Entraîner le modèle sur l'ensemble d'entraînement
    model.fit(diseases_patients_train_model, label_patients_train)

    # Faire des prédictions sur l'ensemble de test
    labels_pred_test = model.predict(diseases_patients_test_model)
    labels_pred_train = model.predict(diseases_patients_train_model)
    proba_test = model.predict_proba(diseases_patients_test_model)[:, 1]
    proba_train = model.predict_proba(diseases_patients_train_model)[:, 1]

    nb_positive_train = np.sum(labels_pred_train==0)
    nb_negative_train = np.sum(labels_pred_train==1)
    nb_positive_test = np.sum(labels_pred_test==0)
    nb_negative_test = np.sum(labels_pred_test==1)


    TP_test = np.sum((label_patients_test==0 )& (labels_pred_test == 0)) / nb_positive_test
    FP_test = np.sum((label_patients_test==1 )& (labels_pred_test == 0)) / nb_positive_test
    TN_test = np.sum((label_patients_test==1 )& (labels_pred_test == 1)) / nb_negative_test
    FN_test = np.sum((label_patients_test== 0)& (labels_pred_test == 1)) / nb_negative_test

    TP_train = np.sum((label_patients_train==0 )& (labels_pred_train == 0)) / nb_positive_train
    FP_train = np.sum((label_patients_train==1 )& (labels_pred_train == 0)) / nb_positive_train
    TN_train = np.sum((label_patients_train==1 )& (labels_pred_train == 1)) / nb_negative_train
    FN_train = np.sum((label_patients_train== 0)& (labels_pred_train == 1)) / nb_negative_train

    accuracy_train = accuracy_score(label_patients_train, labels_pred_train)
    accuracy_test = accuracy_score(label_patients_test, labels_pred_test)

    auc_test = calculate_roc_auc(label_patients_test, proba_test)
    auc_train = calculate_roc_auc(label_patients_train, proba_train)

    proba_avg_zero_test = 1- np.mean(proba_test[label_patients_test==0])
    proba_avg_zero_train = 1- np.mean(proba_train[label_patients_train==0])
    proba_avg_one_test = np.mean(proba_test[label_patients_test==1])
    proba_avg_one_train = np.mean(proba_train[label_patients_train==1])

    print(f'TP_test={TP_test}') 
    print(f'FP_test={FP_test}')
    print(f'TN_test={TN_test}')
    print(f'FN_test={FN_test}')
    print(f'TP_train={TP_train}') 
    print(f'FP_train={FP_train}')
    print(f'TN_train={TN_train}')
    print(f'FN_train={FN_train}')
    print(' ')
    print(f'auc_test={auc_test}')
    print(f'auc_train={auc_train}')
    print(' ')
    print(' ')
    print(f'accuracy_test={accuracy_test}')
    print(f'accuracy_train={accuracy_train}')
    print(' ')
    print(f'proba_avg_zero_test={proba_avg_zero_test}')
    print(f'proba_avg_zero_train={proba_avg_zero_train}')
    print(f'proba_avg_one_test={proba_avg_one_test}')
    print(f'proba_avg_one_train={proba_avg_one_train}')

    return auc_train, auc_test, accuracy_train, accuracy_test, len(diseases_patients_train_model)


dir_chrs = f'{path}codes/Data_Files/Training/SNPS'

path_save_df = f'{path}codes/logs/large/df_scores_decision_tree_{pheno_method}_ld={ld}_keep={keep}_eq={equalized}.csv'

list_files = os.listdir(dir_chrs)
list_chrs = []
for chr in list_files:
    try:
        list_chrs.append(int(chr))
    except:
        pass


##### model constants
columns_score = ['CHR', 'SNP', 'auc_tr', 'auc_te', 'acc_tr', 'acc_te', 'nb_patients']
df_score_SNPS = pd.DataFrame(columns=columns_score)
for CHR in list_chrs:
    dir_SNPS = f'{dir_chrs}/{str(CHR)}'
    list_SNPS = os.listdir(dir_SNPS)
    for SNP in list_SNPS:
        print(SNP, CHR, flush=True)
        try:
            auc_tr, auc_te, acc_tr, acc_te, nb_patients = decision_tree(SNP=SNP,
                            CHR=CHR,
                            method=pheno_method,
                            padding=padding,  
                            pad_token=padding_token, 
                            load_data=load_data, 
                            save_data=save_data, 
                            compute_features=compute_features,
                            data_share=data_share,
                            prop_train_test=prop_train_test,
                            remove_none=remove_none,
                            rollup_depth=rollup_depth,
                            binary_classes=binary_classes,
                            equalized=equalized,
                            interest=interest,
                            keep=keep,
                            scaled=scaled,
                            ld=ld)
            df_scores = pd.DataFrame(columns = columns_score, data=[[CHR, SNP, auc_tr, auc_te, acc_tr, acc_te, nb_patients]])
            df_score_SNPS = pd.concat([df_score_SNPS, df_scores], axis=0, ignore_index=True)
            df_score_SNPS.to_csv(path_save_df)
        except:
            print('this SNP not good', flush=True)
   

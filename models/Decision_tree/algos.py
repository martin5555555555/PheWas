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
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve



def HBGC(dataT, data, labels, equalized, scaled=True,):
    if equalized:
        data, labels = DataTransfo_1SNP.equalize_label(data=data, labels = labels)

    diseases_patients_train_model, diseases_patients_test_model, label_patients_train, label_patients_test = train_test_split(data, labels, test_size = 1-dataT.prop_train_test, random_state=42)

    class_weight = {0: np.sum(label_patients_train == 1) / np.sum(label_patients_train == 0), 1: 1.0}
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

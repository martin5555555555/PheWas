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


embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Paul_Glove/glove_UKBB_omop_rollup_closest_depth_{rollup_depth}_no_1_diseases.pth'
weight_diseases = torch.load(embedding_file_diseases).detach().numpy()[1:]


def cosine_distance(v1, v2):
    # Normalisation des vecteurs
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    
    # Calcul du produit scalaire
    dot_product = np.dot(v1_normalized, v2_normalized)
    
    # La distance cosinus est l'angle entre les deux vecteurs, donc le cosinus de cet angle
    # Nous voulons minimiser la distance, donc 1 - cos(theta)
    cosine_distance = 1 - dot_product
    
    return cosine_distance


distance = []
for pheno_id in range(len(weight_diseases)):
    cosine_distance_par = partial(cosine_distance, weight_diseases[pheno_id])
    distance.append(np.apply_along_axis(cosine_distance_par, arr = weight_diseases, axis=1))         
distance = np.array(distance)

file_save = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Paul_Glove/distance_matrix.py'
np.save(file_save, arr=distance)
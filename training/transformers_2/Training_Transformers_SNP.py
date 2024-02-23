print('lancement programme', flush=True)
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import pandas as pd
import numpy as np
import torch
from functools import partial
import os
import pickle
import time
import copy


from codes.tests.TestsClass import TrainModel, TrainTransformerModel, TestSet
from codes.models.utils import clear_last_line, print_file, number_tests, Unbuffered, plot_infos, plot_ini_infos
from codes.models.data_form.DataForm import PatientList, Patient, DataTransfo_1SNP

train_model =  TrainModel.load_instance_test(4)
indices =None #train_model.dataT.indices_train, train_model.dataT.indices_test

### creation of the reference model
#### framework constants:
model_type = 'transformer'
model_version = 'transformer_V2'
test_name = 'baseline_model_focal'
pheno_method = 'Paul' # Paul, Abby
tryout = False # True if we are doing a tryout, False otherwise 
### data constants:
CHR = 6
SNP = 'rs12203592'
rollup_depth = 4
binary_classes = True #nb of classes related to an SNP (here 0 or 1)
ld = 'no'
vocab_size = None # to be defined with data
padding_token = 0
prop_train_test = 0.8
load_data = False
save_data = True
remove_none = True
compute_features = False
padding = True
list_env_features = []
list_pheno_ids = None #list(np.load('/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/phewas/list_associations_snps/rs673604_paul.npy'))#None
### data format
batch_size = 150
data_share = 1/10#402555
seuil_diseases = 600
equalize_label = True
decorelate = False
threshold_corr = 0.9
threshold_rare = 1000
remove_rare = 'all' # None, 'all', 'one_class'
##### model constants
embedding_method = 'Paul' #None, Paul, Abby
counts_method = 'normal'
freeze_embedding = True
Embedding_size = 4 # Size of embedding.
proj_embed = True
instance_size = 10
n_head = 4 # number of SA heads
n_layer = 2 # number of blocks in parallel
Head_size = 8 # size of the "single Attention head", which is the sum of the size of all multi Attention heads
eval_epochs_interval = 5 # number of epoch between each evaluation print of the model (no impact on results)
eval_batch_interval = 40
p_dropout = 0.2 # proba of dropouts in the model
masking_padding = True # do we include padding masking or not
loss_version = 'cross_entropy' #cross_entropy or focal_loss
L1 = True
gamma = 2
alpha = 63
##### training constants
total_epochs = 10 # number of epochs
learning_rate_max = 0.001 # maximum learning rate (at the end of the warmup phase)
learning_rate_ini = 0.00001 # initial learning rate 
learning_rate_final = 0.0001
warm_up_frac = 0.5 # fraction of the size of the warmup stage with regards to the total number of epochs.
start_factor_lr = learning_rate_ini / learning_rate_max
end_factor_lr = learning_rate_final / learning_rate_max


train_model = TrainTransformerModel(model_version=model_version, test_name=test_name, pheno_method=pheno_method, counts_method=counts_method, tryout=tryout, 
                                    CHR=CHR, SNP=SNP, rollup_depth=rollup_depth, binary_classes=binary_classes, padding_token=padding_token, prop_train_test=prop_train_test,
                                    load_data=load_data,save_data=save_data, remove_none=remove_none, compute_features=compute_features, padding=padding, batch_size=batch_size,
                                    data_share=data_share, seuil_diseases=seuil_diseases, equalize_label=equalize_label, embedding_method=embedding_method, 
                                    freeze_embedding=freeze_embedding, Embedding_size=Embedding_size, n_head=n_head, n_layer=n_layer, Head_size=Head_size,
                                    eval_epochs_interval=eval_epochs_interval, eval_batch_interval=eval_batch_interval, p_dropout=p_dropout, masking_padding=masking_padding,
                                    loss_version=loss_version, gamma=gamma, alpha=alpha, total_epochs=total_epochs, learning_rate_max=learning_rate_max, learning_rate_ini=learning_rate_ini,
                                    learning_rate_final=learning_rate_final,warm_up_frac=warm_up_frac, decorelate=decorelate, threshold_corr=threshold_corr, threshold_rare=threshold_rare,
                                    remove_rare=remove_rare, list_env_features=list_env_features, proj_embed=proj_embed, instance_size=instance_size, indices=indices, list_pheno_ids=list_pheno_ids, L1=L1, ld=ld)


train_model.train_model()



import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import pandas as pd
import numpy as np 

from codes.models.data_form.DataForm import DataTransfo_1SNP, PatientList
import matplotlib.pyplot as plt


### data constants:
CHR = 1
SNP = 'rs673604'
pheno_method = 'Paul' # Paul, Abby
rollup_depth = 4
Classes_nb = 2 #nb of classes related to an SNP (here 0 or 1)
vocab_size = None # to be defined with data
padding_token = 0
prop_train_test = 0.8
load_data = False
save_data = True
remove_none = True
decorelate = True
equalize_label = False
threshold_corr = 0.9
threshold_rare = 1000
remove_rare = 'all' # None, 'all', 'one_class'
compute_features = True
padding = True
list_env_features = []
### data format
batch_size = 20
data_share = 1


dataT = DataTransfo_1SNP(SNP=SNP,
                         CHR=CHR,
                         method=pheno_method,
                         padding=padding,  
                         pad_token=padding_token, 
                         load_data=load_data, 
                         save_data=save_data, 
                         compute_features=compute_features,
                         prop_train_test=prop_train_test,
                         remove_none=True,
                         equalize_label=equalize_label,
                         rollup_depth=rollup_depth,
                         decorelate=decorelate,
                         threshold_corr=threshold_corr,
                         threshold_rare=threshold_rare,
                         remove_rare=remove_rare, 
                         list_env_features=list_env_features,
                         data_share=data_share)

patient_list = dataT.get_patientlist()

import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)
import pandas as pd
import numpy as np 

from codes.models.data_form.DataForm import DataTransfo_1SNP, PatientList
import matplotlib.pyplot as plt

import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)


import numpy as np
import pandas as pd
import vcfpy
import pickle
from functools import partial
from tqdm import tqdm
import os
import time

from codes.models.data_form.DataSets import TabDictDataset
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)


import numpy as np
import pandas as pd
import vcfpy
import pickle
from functools import partial
from tqdm import tqdm
import os
import time

from codes.models.data_form.DataSets import TabDictDataset
from codes.models.data_form.DataForm import DataTransfo_1SNP



### framework constants:
model_type = 'decision_tree'
model_version = 'gradient_boosting'
test_name = '1_test_train_transfo_V1'
pheno_method = 'Abby' # Paul, Abby
tryout = True # True if we are ding a tryout, False otherwise 
### data constants:
### data constants:
CHR = 5
SNP = 'rs16891982'
pheno_method = 'Paul' # Paul, Abby
rollup_depth = 4
ld = 'no'
Classes_nb = 2 #nb of classes related to an SNP (here 0 or 1)
vocab_size = None # to be defined with data
padding_token = 0
prop_train_test = 0.8
load_data = True
save_data = False
remove_none = True
decorelate = False
equalize_label = False
threshold_corr = 0.9
threshold_rare = 50
remove_rare = 'all' # None, 'all', 'one_class'
compute_features = True
padding = False
list_env_features = []
list_pheno_ids = None #list(np.load(f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/phewas/list_associations_snps/{SNP}_paul.npy'))

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
                         remove_none=remove_none,
                         equalize_label=equalize_label,
                         rollup_depth=rollup_depth,
                         decorelate=decorelate,
                         threshold_corr=threshold_corr,
                         threshold_rare=threshold_rare,
                         remove_rare=remove_rare, 
                         list_env_features=list_env_features,
                         data_share=data_share,
                         list_pheno_ids=list_phenos_ids,
                         ld=ld)
#patient_list = dataT.get_patientlist()
data, labels, indices_env, name_envs = dataT.get_tree_data(with_env=False, load_possible=False, only_relevant=True)

file = '/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Pheno/Paul/ukbb_omop_rolled_up_depth_4_closest_ancestor_tree_counts.npy'
df_tot.to_csv(file)
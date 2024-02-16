print('begining of regression', flush=True)
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import numpy as np
import pandas as pd
import torch
import pickle
import time
import os
import tensorboard

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from tqdm.auto import tqdm
import statsmodels.api as sm
from multiprocessing import Pool

from codes.models.utils import number_tests, Unbuffered
from codes.models.data_form.DataForm import DataTransfo_1SNP


### framework constants:
model_type = 'logistic_regression'
model_version = 'global'
test_name = 'train_log_reg_paul'
pheno_method = 'Paul' # Paul, Abby
tryout = False # True if we are ding a tryout, False otherwise 
### data constants:
CHR = 1
SNP = 'rs673604'
rollup_depth = 4
Classes_nb = 2 #nb of classes related to an SNP (here 0 or 1)
vocab_size = None # to be defined with data
padding_token = 0
prop_train_test = 0.8
load_data = True
save_data = False
remove_none = True
compute_features = True
padding = True
share = 0.1
seuil_diseases = 50
equalize_label = False
### data format
batch_size = 20
data_share = 1

##### model constants
p_value_treshold = 1e-4

##### training constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_processes = 60


path = '/gpfs/commons/groups/gursoy_lab/mstoll/codes/'

#check test name
model_dir = path + f'logs/SNPS/{str(CHR)}/{SNP}/{model_type}/{model_version}/{pheno_method}'
os.makedirs(model_dir, exist_ok=True)
#check number tests
number_test = number_tests(model_dir)
test_name_with_infos = str(number_test) + '_' + test_name + 'tryout'*tryout
test_dir = f'{model_dir}/{test_name_with_infos}/'
log_data_dir = f'{test_dir}/data/'
log_res_dir = f'{test_dir}/res/'
log_slurm_outputs_dir = f'{test_dir}/Slurm/Outputs/'
log_slurm_errors_dir = f'{test_dir}/Slurm/Errors/'

os.makedirs(log_data_dir)
os.makedirs(log_res_dir)
os.makedirs(log_slurm_outputs_dir)
os.makedirs(log_slurm_errors_dir)


log_data_path_pickle = f'{test_dir}/data/{test_name}.pkl'


log_res_path_pickle = f'{test_dir}/res/{test_name}.pkl'
log_slurm_outputs_path = f'{test_dir}/Slurm/Outputs/{test_name}.pth'
log_slurm_error_path = f'{test_dir}/Slurm/Errors/{test_name}.pth'

   
# Redirect  output to a file
sys.stdout = open(log_slurm_outputs_path, 'w')
sys.stderr = open(log_slurm_error_path, 'w')

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

print('creation of data')

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
                         seuil_diseases=seuil_diseases,
                         equalize_label=equalize_label)
patient_list = dataT.get_patientlist()

nb_diseases = patient_list.nb_distinct_diseases_tot
pheno_data, label_data = patient_list.get_tree_data()
pheno_data = np.array(pheno_data)


def logreg(pheno_data, label_data):
    try:
        logit_model = sm.Logit(label_data, pheno_data)
        result = logit_model.fit(disp=False)
        p_value = result.pvalues[0]
        coeff = result.params[0]
        return p_value, coeff, result
    except:
        return None, None, None

print('begining training')
pheno_indexs = np.arange(nb_diseases)
pheno_indexs_splits = np.array_split(pheno_indexs, nb_diseases // num_processes )

## Open tensor board writer
output_file = log_slurm_outputs_path
with open(output_file, 'w') as file:
    file.truncate()
    file.close()

    start_time_cell = time.time()
p_values = np.zeros(nb_diseases)
coeffs = np.zeros(nb_diseases)
models = []
label_data_train = [np.array(label_data) for _ in range(num_processes)]
for batch_phenos_index in tqdm(pheno_indexs_splits, desc="batch num"):
    start_time =  time.time()

    if len(batch_phenos_index) == num_processes:
        pool = Pool(processes=num_processes)
        pheno_data_train = [pheno_data[:, pheno_index] for pheno_index in batch_phenos_index]



        results = pool.starmap(logreg, zip(pheno_data_train, label_data_train))
        # Fermez le pool de processus
        pool.close()
        pool.join()
        p_values[batch_phenos_index]= np.array([result[0] for result in results])
        coeffs[batch_phenos_index] = np.array([result[1] for result in results])
        for result in results:
            models.append(result[2])

    else:
        for pheno_index in batch_phenos_index:
            pheno_data_train = pheno_data[:, pheno_index]
            p_value, coeff, model = logreg(pheno_data_train, label_data_train)
            p_values[pheno_index] = p_value
            coeffs[pheno_index] = coeff
            models.append(model)
    print(f'batch finished in {time.time() - start_time} s')
print(f'program over in {time.time()-start_time_cell}s')


with open(log_data_path_pickle, 'wb') as file:
    pickle.dump(patient_list, file)
print('Data saved to %s' % log_data_path_pickle)

res_dic = {
    'model': models, 
    'p_values' : p_values,
    'coeffs' : coeffs
}

with open(log_res_path_pickle, 'wb') as file:
    pickle.dump(res_dic, file)



print('Model saved to %s' % log_res_path_pickle)
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import torch
from functools import partial
import os

from codes.models.utils import clear_last_line, print_file, number_tests, Unbuffered, plot_infos, plot_ini_infos



##############################################################defining framework constants ###################################

#### framework constants:
model_type = 'transformer'
model_version = 'transformer_V2'
test_name = 'test_balance_small_data_abby_balanced'
pheno_method = 'Abby' # Paul, Abby
tryout = False # True if we are doing a tryout, False otherwise 
### data constants:
CHR = 1
SNP = 'rs673604'
rollup_depth = 4
Classes_nb = 2 #nb of classes related to an SNP (here 0 or 1)
vocab_size = None # to be defined with data
padding_token = 0
prop_train_test = 0.8
load_data = False
save_data = False
remove_none = True
compute_features = False
padding = True
### data format
batch_size = 20
data_share = 1 #402555
seuil_diseases = 600
equalize_label = False

##### model constants
embedding_method = 'Paul' #None, Paul, Abby
freeze_embedding = False
Embedding_size = 100 # Size of embedding.
n_head = 5 # number of SA heads
n_layer = 3 # number of blocks in parallel
Head_size = 150  # size of the "single Attention head", which is the sum of the size of all multi Attention heads
eval_epochs_interval = 5 # number of epoch between each evaluation print of the model (no impact on results)
eval_batch_interval = 40
p_dropout = 0.1 # proba of dropouts in the model
masking_padding = True # do we include padding masking or not
loss_version = 'focal_loss' #cross_entropy or focal_loss
gamma = 2
##### training constants
total_epochs = 100 # number of epochs
learning_rate_max = 0.01 # maximum learning rate (at the end of the warmup phase)
learning_rate_ini = 0.00001 # initial learning rate 
learning_rate_final = 0.0001
warm_up_frac = 0.5 # fraction of the size of the warmup stage with regards to the total number of epochs.
start_factor_lr = learning_rate_ini / learning_rate_max
end_factor_lr = learning_rate_final / learning_rate_max
warm_up_size = int(total_epochs*warm_up_frac)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lr_lambda(current_epoch, warm_up_size=warm_up_size): ## function that defines the evolution of the learning rate.
    warm_up_size = int(total_epochs*warm_up_frac)
    if current_epoch < warm_up_size:
        return learning_rate_ini + current_epoch*(learning_rate_max - learning_rate_ini) / warm_up_size
    else:
        return learning_rate_max / (current_epoch - warm_up_size + 1)
lr_lambda = partial(lr_lambda, warm_up_size=warm_up_size) 

 

#######################################################Link toward directories ###############################################################################################
path = '/gpfs/commons/groups/gursoy_lab/mstoll/codes/'

#check test name
model_dir = path + f'logs/SNPS/{str(CHR)}/{SNP}/{model_type}/{model_version}/{pheno_method}'
os.makedirs(model_dir, exist_ok=True)
#check number tests
number_test = number_tests(model_dir)
test_name_with_infos = str(number_test) + '_' + test_name + 'tryout'*tryout
test_dir = f'{model_dir}/{test_name_with_infos}/'
log_model_dir = f'{test_dir}/model/'
log_data_dir = f'{test_dir}/data/'
log_info_dir = f'{test_dir}/infos/tensorboard/'
log_slurm_outputs_dir = f'{test_dir}/Slurm/Outputs/'
log_slurm_errors_dir = f'{test_dir}/Slurm/Errors/'

os.makedirs(log_model_dir)
os.makedirs(log_data_dir)
os.makedirs(log_info_dir)
os.makedirs(log_slurm_outputs_dir)
os.makedirs(log_slurm_errors_dir)


log_model_path_torch = f'{test_dir}/model/{test_name}.pth'
log_model_path_pickle = f'{test_dir}/model/{test_name}.pkl'
log_data_path_pickle = f'{test_dir}/data/{test_name}.pkl'


log_info_path = f'{test_dir}/infos/tensorboard/{test_name}'
log_slurm_outputs_path = f'{test_dir}/Slurm/Outputs/{test_name}.pth'
log_slurm_error_path = f'{test_dir}/Slurm/Errors/{test_name}.pth'

   
# Redirect  output to a file
sys.stdout = open(log_slurm_outputs_path, 'w')
sys.stderr = open(log_slurm_error_path, 'w')
sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

print(f'Begining of training program, device={device}')


import numpy as np
import pandas as pd
import torch.nn as nn
import pickle
import time
import shutil

import tensorboard

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from rich.progress import Progress
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, accuracy_score


from codes.models.data_form.DataForm import DataTransfo_1SNP
from codes.models.Transformers.Embedding import EmbeddingPheno
from codes.models.Transformers.dic_model_versions import DIC_MODEL_VERSIONS


print(f'Done with imports')

print( f'cuda availbale : {torch.cuda.is_available()}')




######################################### creating the data ####################################################
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
                         equalize_label=equalize_label,
                         seuil_diseases=seuil_diseases)
patient_list = dataT.get_patientlist()




indices_train, indices_test = dataT.get_indices_train_test(patient_list=patient_list,prop_train_test=prop_train_test)
patient_list_transformer_train, patient_list_transformer_test = patient_list.get_transformer_data(indices_train.astype(int), indices_test.astype(int))
#creation of torch Datasets:
dataloader_train = DataLoader(patient_list_transformer_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(patient_list_transformer_test, batch_size=batch_size, shuffle=True)




if patient_list.nb_distinct_diseases_tot==None:
    vocab_size = patient_list.get_nb_distinct_diseases_tot()
if patient_list.nb_max_counts_same_disease==None:
    max_count_same_disease = patient_list.get_max_count_same_disease()
max_count_same_disease = patient_list.nb_max_counts_same_disease
vocab_size = patient_list.nb_distinct_diseases_tot

print(f'\n vocab_size : {vocab_size}, max_count : {max_count_same_disease}\n', 
      f'length_patient = {patient_list.get_nb_max_distinct_diseases_patient()}\n',
      f'sparcity = {patient_list.sparsity}\n',
      f'nombres patients  = {len(patient_list)}')

####################################################### creation of the model #######################################################
Embedding  = EmbeddingPheno(method=embedding_method, vocab_size=vocab_size, max_count_same_disease=max_count_same_disease, Embedding_size=Embedding_size, rollup_depth=rollup_depth, freeze_embed=freeze_embedding)

### creation of the model
### creation of the model
ClassModel = DIC_MODEL_VERSIONS[model_version]
model = ClassModel(pheno_method = pheno_method,
                             Embedding = Embedding,
                             Head_size=Head_size,
                             Classes_nb=Classes_nb,
                             n_head=n_head,
                             n_layer=n_layer,
                             mask_padding=masking_padding, 
                             padding_token=0, 
                             p_dropout=p_dropout, 
                             loss_version = loss_version,
                             gamma = gamma)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_max)
lr_scheduler_warm_up = LinearLR(optimizer, start_factor=start_factor_lr , end_factor=1, total_iters=warm_up_size, verbose=False) # to schedule a modification in the learning rate
lr_scheduler_final = LinearLR(optimizer, start_factor=1, total_iters=total_epochs-warm_up_size, end_factor=end_factor_lr)
lr_scheduler = SequentialLR(optimizer, schedulers=[lr_scheduler_warm_up, lr_scheduler_final], milestones=[warm_up_size])


output_file = log_slurm_outputs_path
            ## Open tensor board writer
writer = SummaryWriter(log_tensorboard_path)
dic_features_list = {
            'list_training_loss' : [],
            'list_validation_loss' : [],
            'list_proba_avg_zero' : [],
            'list_proba_avg_one' : [],
            'list_auc_validation' : [],
            'list_accuracy_validation' : [],
            'list_f1_validation' : [],
            'epochs' : [] }
# Training Loop
start_time_training = time.time()
print_file(output_file, f'Beginning of the program for {total_epochs} epochs', new_line=True)
# Training Loop
plot_ini_infos(model, output_file, dataloader_test, dataloader_train, writer, dic_features_list)
for epoch in range(1, total_epochs+1):

    start_time_epoch = time.time()
    total_loss = 0.0  
    
    #with tqdm(total=len(dataloader_train), position=0, leave=True) as pbar:
    for k, (batch_sentences, batch_counts, batch_labels) in enumerate(dataloader_train):

        # evaluate the loss
        logits, loss = model(batch_sentences, batch_counts, batch_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    

        total_loss += loss.item()

        optimizer.step()

        if k %eval_batch_interval == 0:
            clear_last_line(output_file)
            print_file(output_file, f'Progress in batch = {round(k / len(dataloader_train)*100, 2)} %, time batch : {time.time() - start_time_epoch}', new_line=False)

    if epoch % eval_epochs_interval == 0:
        dic_features = plot_infos(model, output_file, epoch, total_loss, start_time_epoch, dataloader_train, dataloader_test, optimizer, writer, dic_features_list, plots_path)

    
    
    lr_scheduler.step()

model = model.to('cpu')

torch.save(model.state_dict(), log_model_path_torch)
print('Model saved to %s' % log_model_path_torch)
# Print time
print_file(output_file, f"Training finished: {int(time.time() - start_time_training)} seconds", new_line=True)
start_time = time.time()

with open(log_model_path_pickle, 'wb') as file:
    pickle.dump(model, file)
print('Model saved to %s' % log_model_path_pickle)

dic_data = {
    'patient_list':patient_list,
    'data' : dataT
}
with open(log_data_path_pickle, 'wb') as file:
    pickle.dump(dic_data, file)
print('Data saved to %s' % log_data_path_pickle)





## Add hyper parameters to tensorboard
hyperparams = {"CHR": CHR, "SNP": SNP, "ROLLUP LEVEL": rollup_depth,
               'PHENO_METHOD':pheno_method, 'EMBEDDING_METHOD':embedding_method,
              'EMBEDDING SIZE': Embedding_size, 'ATTENTION HEADS': n_head, 'BLOCKS': n_layer,
              'LR':1 , 'DROPOUT': p_dropout, 'NUM_EPOCHS': total_epochs, 
              'BATCH_SIZE': batch_size, 
              'PADDING_MASKING':masking_padding,
              'VERSION': model_version,
              'NB_Patients' : len(patient_list),
              'LOSS_VERSION' : loss_version,

            }
metrics = {'loss': val_loss, "AUC Score": auc_score, "F1 Score": f1, "Accuracy": accuracy}
writer.add_hparams(hyperparams, metrics)
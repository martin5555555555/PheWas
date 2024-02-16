import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import os
import pandas as pd
import numpy as np 
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter


from codes.models.data_form.DataForm import DataTransfo_1SNP, PatientList
from codes.models.metrics import calculate_roc_auc, calculate_classification_report, calculate_loss, get_proba
from codes.models.Generative.Embeddings import EmbeddingPheno, EmbeddingSNPS
from codes.models.Generative.GenerativeModel import GenerativeModelPheWasV1
from codes.models.utils import print_file, plot_infos, plot_ini_infos, clear_last_line
from sklearn.metrics import f1_score, accuracy_score


import matplotlib.pyplot as plt


### data constants:
model_type = 'Generative_Transformer'
model_version = 'V1'
test_name = 'tests_generative_1'
CHR = 1
SNP = 'rs673604'
pheno_method = 'Abby' # Paul, Abby
rollup_depth = 4
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
list_env_features = ['age', 'sex']
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
#patient_list = dataT.get_patientlist()
patient_list = dataT.get_patientlist()
patient_list.unpad_data()


rollup_depth = 4
Head_size_pheno = 4
n_head_pheno = 2
n_layer_pheno = 2
instance_size_pheno = 10
Embedding_size_pheno = 10
embedding_method_pheno = None
proj_embed_pheno = False
freeze_embed_pheno = False
loss_version_pheno = 'cross_entropy'
p_dropout = 0.1
device = 'cpu'
pheno_method = 'Abby'
embedding_method_pheno = None
embedding_method_SNPS = None
freeze_embed_SNPS = False
nb_phenos = patient_list.get_nb_distinct_diseases_tot()
nb_SNPS = 2
Embedding_size_SNPS = 10
n_head_SNPS = 2
Head_size_SNPS = 4
loss_version_SNPS = 'cross_entropy'
n_layer_SNPS = 2
instance_size_SNPS = 10
mask_padding = True
#multi
n_head_cross = 2
Head_size_cross = 4
n_layer_cross = 2
instance_size_cross = 10

nb_phenos_possible = patient_list.get_nb_distinct_diseases_tot()
vocab_size = nb_phenos_possible + 1 # masking
##### training constants
total_epochs = 10# number of epochs
learning_rate_max = 0.001 # maximum learning rate (at the end of the warmup phase)
learning_rate_ini = 0.00001 # initial learning rate 
learning_rate_final = 0.0001
warm_up_frac = 0.5 # fraction of the size of the warmup stage with regards to the total number of epochs.
start_factor_lr = learning_rate_ini / learning_rate_max
end_factor_lr = learning_rate_final / learning_rate_max
warm_up_size = int(warm_up_frac*total_epochs)
padding_masking = True

eval_batch_interval = 40
eval_epochs_interval = 1

#################### generate the ouptut files and dirs ############################################
path = '/gpfs/commons/groups/gursoy_lab/mstoll/codes/'
#check test name
model_dir = path + f'logs/runs/SNPS/{str(CHR)}/{SNP}/{model_type}/{model_version}/{pheno_method}'
model_plot_dir = path + f'logs/plots/tests/SNP/{str(CHR)}/{SNP}/{model_type}/{model_version}/{pheno_method}/'

os.makedirs(model_dir, exist_ok=True)
os.makedirs(model_plot_dir, exist_ok=True)
#check number tests
test_dir = f'{model_dir}/{test_name}/'
print(test_dir)
log_data_dir = f'{test_dir}/data/'
log_tensorboard_dir = f'{test_dir}/tensorboard/'
log_slurm_outputs_dir = f'{test_dir}/Slurm/Outputs/'
log_slurm_errors_dir = f'{test_dir}/Slurm/Errors/'
os.makedirs(log_data_dir, exist_ok=True)
os.makedirs(log_tensorboard_dir, exist_ok=True)
os.makedirs(log_slurm_outputs_dir, exist_ok=True)
os.makedirs(log_slurm_errors_dir, exist_ok=True)


log_data_path_pickle = f'{test_dir}/data/{test_name}.pkl'
log_tensorboard_path = f'{test_dir}/tensorboard/{test_name}'
log_slurm_outputs_path = f'{test_dir}/Slurm/Outputs/{test_name}.txt'
log_slurm_error_path = f'{test_dir}/Slurm/Errors/{test_name}.txt'
model_plot_path = path + f'logs/plots/tests/SNP/{str(CHR)}/{SNP}/{model_type}/{model_version}/{pheno_method}/{test_name}.png'

############ generate the masked list of diseases #############################################
start_time = time.time()
print('generating the data files')
list_pheno_truth = []
list_labels = []
list_diseases_sentence_masked = []
for patient in patient_list:
    diseases_sentence = torch.tensor(patient.diseases_sentence)
    nb_diseases = len(diseases_sentence)
    masks = np.zeros((nb_diseases, nb_diseases)).astype(bool)
    np.fill_diagonal(masks,True)
    diseases_sentence_masked = np.tile(diseases_sentence, nb_diseases).reshape(nb_diseases, nb_diseases)
    pheno_Truth = diseases_sentence_masked[masks]
    labels = [np.array([patient.SNP_label])]*nb_diseases
    diseases_sentence_masked[masks] = nb_phenos 

    list_pheno_truth.extend(pheno_Truth)
    list_labels.extend(labels)
    list_diseases_sentence_masked.extend(diseases_sentence_masked)
print(f'generated files in {time.time() - start_time} seconds')

################################### padding the data ###################################################
list_diseases_new = []
nb_max_distinct_diseases_patient= patient_list.get_nb_max_distinct_diseases_patient() 
for list_diseases in list_diseases_sentence_masked:
    padd = np.zeros(nb_max_distinct_diseases_patient- len(list_diseases), dtype=int)
    list_diseases_new.append(np.concatenate([list_diseases, padd]).astype(int))
list_diseases_sentence_masked = list_diseases_new


list_data_gen = list(zip(list_diseases_sentence_masked, list_pheno_truth, list_labels))
indices= np.arange(len(list_data_gen))
np.random.shuffle(indices)
indices_train= indices[:int(prop_train_test * len(list_data_gen))]
indices_test = indices[int(prop_train_test * len(list_data_gen)):]


data_training = [list_data_gen[i] for i in indices_train]
data_test = [list_data_gen[i] for i in indices_test]


dataloader_train = DataLoader(data_training, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)



Embedding_pheno = EmbeddingPheno(method=embedding_method_pheno, vocab_size=vocab_size, Embedding_size=Embedding_size_pheno,
     rollup_depth=rollup_depth, freeze_embed=freeze_embed_pheno, dicts=dataT.dicts)

Embedding_SNPS = EmbeddingSNPS(method=embedding_method_SNPS, nb_SNPS=nb_SNPS, Embedding_size=Embedding_size_SNPS, freeze_embed=freeze_embed_SNPS)
    

model = GenerativeModelPheWasV1(n_head_pheno=n_head_pheno, Head_size_pheno=Head_size_pheno, Embedding_pheno=Embedding_pheno, Embedding_SNPS=Embedding_SNPS,
    instance_size_pheno=instance_size_pheno, n_layer_pheno=n_layer_pheno,  nb_SNPS=nb_SNPS, n_layer_SNPS=n_layer_SNPS, n_head_SNPS=n_head_SNPS, mask_padding=mask_padding,
    Head_size_SNPS=Head_size_SNPS, instance_size_SNPS=instance_size_SNPS, nb_phenos_possible=nb_phenos_possible,
    n_head_cross=n_head_cross, Head_size_cross=Head_size_cross, n_layer_cross=n_layer_cross, p_dropout=p_dropout, device=device,
    loss_version_pheno=loss_version_pheno, loss_version_SNPS=loss_version_SNPS, gamma=2, alpha=1, padding_token=padding_token)



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_max)
lr_scheduler_warm_up = LinearLR(optimizer, start_factor=start_factor_lr , end_factor=1, total_iters=warm_up_size, verbose=False) # to schedule a modification in the learning rate
lr_scheduler_final = LinearLR(optimizer, start_factor=1, total_iters=total_epochs-warm_up_size, end_factor=end_factor_lr)
lr_scheduler = SequentialLR(optimizer, schedulers=[lr_scheduler_warm_up, lr_scheduler_final], milestones=[warm_up_size])


######################################################## Training Loop ###################################################
output_file = log_slurm_outputs_path
writer = SummaryWriter(log_tensorboard_path)

## Open tensor board writer
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
    for k, (batch_sentences_pheno, batch_labels_pheno, batch_sentences_SNPS) in enumerate(dataloader_train):
        start_time_batch = time.time()
        
        batch_sentences_pheno = batch_sentences_pheno.to(device)
        batch_labels_pheno = batch_labels_pheno.to(device)
        batch_sentences_SNPS = batch_sentences_SNPS.to(device)

        # evaluate the loss
        logits, loss = model(batch_sentences_pheno, batch_sentences_SNPS,value='pheno', targets= batch_labels_pheno)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    

        total_loss += loss.item()

        optimizer.step()

        if k % eval_batch_interval == 0:
            clear_last_line(output_file)
            print_file(output_file, f'Progress in epoch {epoch}  = {round(k / len(dataloader_train)*100, 2)} %, time batch : {time.time() - start_time_epoch}', new_line=False)

    if epoch % eval_epochs_interval == 0:
        dic_features = plot_infos(model, output_file, epoch, total_loss, start_time_epoch, dataloader_train, dataloader_test, optimizer, writer, dic_features_list, model_plot_path)

    
    
    lr_scheduler.step()

dic_features = dic_features
model.to('cpu')
#model.write_embedding(writer)
# Print time
print_file(output_file, f"Training finished: {int(time.time() - start_time_training)} seconds", new_line=True)
start_time = time.time()





## Add hyper parameters to tensorboard
hyperparams = {"CHR" : CHR, "SNP" : SNP, "ROLLUP LEVEL" : rollup_depth,
            'PHENO_METHOD': pheno_method, 'EMBEDDING_METHOD': embedding_method_pheno,
            'EMBEDDING SIZE' : Embedding_size_pheno, 'ATTENTION HEADS' : n_head_pheno, 'BLOCKS' : n_layer_pheno,
            'LR':1 , 'DROPOUT' : p_dropout, 'NUM_EPOCHS' : total_epochs, 
            'BATCH_SIZE' : batch_size, 
            'PADDING_MASKING': padding_masking,
            'VERSION' : model_version,
            'NB_Patients'  : len(patient_list),
            'LOSS_VERSION'  : loss_version_pheno,
            }

writer.add_hparams(hyperparams, dic_features)



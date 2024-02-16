import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from codes.models.metrics import calculate_roc_auc, get_proba
class NaiveModelWeights(nn.Module):
    def __init__(self, pheno_nb):
        super().__init__()
        
        self.linear_weights_predictor = nn.Linear(pheno_nb, pheno_nb,bias=True,  dtype=float)
        self.logits_predictor = nn.Linear(pheno_nb, 2 *pheno_nb, bias=True, dtype=float)

    def forward(self, x, labels_target=None):
        B, P = x.shape
        weights = self.linear_weights_predictor(x)
        prob_weights = F.softmax(weights).view(B, P, 1)
        logits = self.logits_predictor(x).view(B, P, 2)
        logits = (logits.transpose(1, 2)) @ prob_weights
        

        if labels_target != None:
            err = F.cross_entropy(logits, labels_target.view(B, 1))#torch.sqrt(torch.sum((pred_probas - labels_target)**2)/len(x)) 
        return logits, err


    def evaluate(self, dataloader_test):
        self.eval()
        print('beginning inference evaluation')
        start_time_inference = time.time()
        predicted_labels_list = []
        predicted_probas_list = []
        true_labels_list = []

        total_loss = 0.
        self.eval()
        with torch.no_grad():
            for batch_data in dataloader_test:
                data_train= batch_data['data']
                labels_train = batch_data['label']


                logits, loss = self(data_train, labels_train)
                total_loss += loss.item()
                predicted_probas = F.softmax(logits, dim=1).detach().numpy()
                predicted_labels = (predicted_probas[:, 1] > 0.5).astype(int)

                #predicted_labels = self.predict(batch_sentences, batch_counts)
                predicted_labels_list.extend(predicted_labels)
                predicted_probas_list.extend(predicted_probas)
                true_labels_list.extend(labels_train)
        f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')
        accuracy = accuracy_score(true_labels_list, predicted_labels_list)
        auc_score = calculate_roc_auc(true_labels_list, np.array(predicted_probas_list)[:, 1], return_nan=True)
        proba_avg_zero, proba_avg_one = get_proba(true_labels_list, predicted_probas_list)
        
        return f1, accuracy, auc_score, total_loss / len(dataloader_test), proba_avg_zero, proba_avg_one, predicted_probas_list, true_labels_list

class CustomDatasetWithLabels(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample
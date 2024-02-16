import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report
from torch.nn import functional as F

def calculate_roc_auc(y_true, predicted_probabilities, return_nan=False, multi_class=None):
    # Check the number of unique classes
    num_classes = len(np.unique(y_true))
    print(f'num classes = {num_classes}')
    # Check if there is more than one class
    if num_classes > 1:
        if multi_class == None:
        # Compute ROC-AUC score
            roc_auc = roc_auc_score(y_true, predicted_probabilities)
        else:
            roc_auc = roc_auc_score(y_true, predicted_probabilities, multi_class=multi_class)

        return roc_auc
    else:
        # Return NaN if there is only one class
        if return_nan:
            return np.nan
        else:
            print("Only one class present in y_true. ROC AUC score is not defined in that case.")

def calculate_classification_report(true_labels_list, predicted_labels_list, return_nan=True):
    # Check the number of unique classes
    num_classes = len(np.unique(true_labels_list))

    # Check if there is more than one class
    if num_classes > 1:
        # Compute ROC-AUC score
        report = classification_report(true_labels_list, predicted_labels_list)
        return report
    else:
        # Return NaN if there is only one class
        if return_nan:
            return np.nan
        else:
            print("Only one class present in y_true. ROC AUC score is not defined in that case.")

def get_proba(true_labels_list, predicted_probas_list):
    avg_proba_zero = np.mean(np.array(predicted_probas_list)[:,0][np.array(true_labels_list)==0])
    avg_proba_one = np.mean(np.array(predicted_probas_list)[:,1][np.array(true_labels_list)==1])
    return avg_proba_zero, avg_proba_one

def calculate_loss(logits, logits_relevant, targets=None, loss_type='cross_entropy', gamma=None, alpha=None, L1=True):
    if targets is None:
            loss = None
    else:
        #target : shape B,
        
        if loss_type == 'cross_entropy':
            cross_entropy = F.cross_entropy(logits, targets)
            loss = cross_entropy
        elif loss_type == 'focal_loss':
            alphas = ((1 - targets) * (alpha-1)).to(torch.float) + 1
            probas = F.softmax(logits)
            certidude_factor =  (1-probas[[range(probas.shape[0]), targets]])**gamma * alphas
            cross_entropy = F.cross_entropy(logits, targets, reduce=False)
            loss = torch.dot(cross_entropy, certidude_factor)
        elif loss_type == 'predictions':
            probas = F.softmax(logits)
            predictions = (probas[:,1] > 0.5).to(int)
            loss = torch.sum((predictions-targets)**2)/len(predictions)
        
        if L1:
            loss_l1 = torch.norm(logits_relevant, p=1)
            return loss + loss_l1
        else:
            return loss
import numpy as np

def get_risk_pheno(data, labels, pheno_nb):
    labels_ac = labels[data[:,pheno_nb]==1]
    labels_deac = labels[data[:,pheno_nb]==0]
    proba_mut_ac = np.sum(labels_ac==1)/len(labels_ac)
    proba_mut_deac = np.sum(labels_deac==1)/len(labels_deac)
    ratio  = proba_mut_ac / proba_mut_deac
    return max([ratio ,ratio**-1])
def get_pred_naive(data, labels, pheno_nb, proba=False):
    labels_ac = labels[data[:,pheno_nb]==1]
    nb_ones_ac = np.sum(labels_ac==1)
    nb_zeros_ac = np.sum(labels_ac==0)
    proba = nb_zeros_ac / len(labels_ac)
    label = (1 if nb_ones_ac > nb_zeros_ac else 0)
    return proba, label

def get_pred_sentence(probas_pred_naive, labels_pred_naive, sentence, method='max'):
    sentence = sentence.astype(bool)
    labels_naive = labels_pred_naive[sentence].astype(bool)
    probas_naive = probas_pred_naive[sentence].astype(bool)

    if method=='mean':
        if np.mean(probas_naive)>0.5:
            return 1
        else:
            return 0
    if method=='max':
        argmax = np.argmax((probas_naive-0.5)**2)
        return labels_naive[argmax]
    
    
def get_risk_pheno(data, labels, pheno_nb):    
    labels_1 = labels[data[:,pheno_nb]==1]
    labels_0 = labels[data[:,pheno_nb]==0]
    P0 = np.sum(labels_0==1)/len(labels_0)
    P1 = np.sum(labels_1==1)/len(labels_1)
    F0 = max(P0, 1-P0)
    F1 = max(P1, 1-P1)
    return F0, F1
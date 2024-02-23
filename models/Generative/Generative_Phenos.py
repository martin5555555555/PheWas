import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import time
from sklearn.metrics import f1_score, accuracy_score
from codes.models.metrics import calculate_roc_auc, calculate_classification_report, calculate_loss, get_proba


class PredictLogit(nn.Module):
    def __init__(self, instance_size_pheno, nb_phenos_possible):
        super().__init__()
        self.ln2_phenos = nn.LayerNorm(instance_size_pheno)
        self.get_logits_phenos = nn.Linear(instance_size_pheno, nb_phenos_possible)
    
    def forward(self, pheno_sentence):
        pheno_sentence = self.ln2_phenos(pheno_sentence)
        logits = self.get_logits_phenos(pheno_sentence)
        logits_mean = logits.mean(axis=1)
        return logits_mean

class PhenotypeEncodingAlone(nn.Module):
    def __init__(self, Embedding, Head_size, n_head, n_layer, mask_padding=False, padding_token=None, p_dropout=0, device='cpu', instance_size=None, proj_embed=True, loss_version='cross_entropy', gamma=2, alpha=1, nb_phenos_possible=0):
        super().__init__()
       
        self.Embedding_size = Embedding.Embedding_size
        self.instance_size=instance_size
        self.nb_phenos_possible = nb_phenos_possible
        self.proj_embed = proj_embed
        if not self.proj_embed:
            self.instance_size = self.Embedding_size
        if self.proj_embed:
            self.projection_embed = nn.Linear(self.Embedding_size, self.instance_size)
        self.blocks =PadMaskSequential(*[BlockPheno(self.instance_size, n_head=n_head, Head_size=Head_size, p_dropout=p_dropout) for _ in range(n_layer)]) #Block(self.instance_size, n_head=n_head, Head_size=Head_size) 
        self.ln_f = nn.LayerNorm(self.instance_size) # final layer norm
        self.predict_logit = PredictLogit(instance_size_pheno=instance_size, nb_phenos_possible=nb_phenos_possible-1) #-1 for padding

        self.Embedding = Embedding
        self.mask_padding = mask_padding
        self.padding_token = padding_token
        self.padding_mask = None
        self.device = device
        self.loss_version = loss_version
        self.gamma = gamma
        self.alpha = alpha
       
        self.diseases_embedding_table = Embedding.distinct_diseases_embeddings
        #if self.pheno_method == 'Paul':
        # self.counts_embedding_table = Embedding.counts_embeddings

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            

    def create_padding_mask(self, diseases_sentence):
        B, S = np.shape(diseases_sentence)[0], np.shape(diseases_sentence)[1]
        mask = torch.where(diseases_sentence==self.padding_token)
        padding_mask_mat = torch.ones((B, S, S), dtype=torch.bool)
        padding_mask_mat[mask] = 0
        padding_mask_mat.transpose(-2,-1)[mask] = 1

        padding_mask_probas = torch.zeros((B, S)).to(bool)
        padding_mask_probas[mask] = True
        padding_mask_probas = padding_mask_probas.view(B, S)
        return padding_mask_mat.to(self.device), padding_mask_probas # 1 if masked, 0 else

    def set_padding_mask_transformer(self, padding_mask, padding_mask_probas):
        self.padding_mask = padding_mask
        self.padding_mask_probas = padding_mask_probas
    
    def forward(self, diseases_sentence, targets=None):
        Batch_len, Sentence_len = diseases_sentence.shape

        diseases_sentence = diseases_sentence.to(self.device)
        #counts_diseases = counts_diseases.to(self.device)
        
        if self.mask_padding:
            padding_mask, padding_mask_probas = self.create_padding_mask(diseases_sentence)
            self.set_padding_mask_transformer(padding_mask, padding_mask_probas)
            #self.blocks.set_padding_mask_sequential(self.padding_mask)

        diseases_sentences_embedded = self.diseases_embedding_table(diseases_sentence) # shape B, S, E

        x = diseases_sentences_embedded 
    
        #if self.pheno_method == 'Paul':
        #    counts_diseases_embedded = self.counts_embedding_table(counts_diseases) # shape B, S, E
        #    #x = x + counts_diseases_embedded # shape B, S, E 
        
        if self.proj_embed:
            x = self.projection_embed(x)
        x = self.blocks(x, padding_mask) # shape B, S, E
        
        logits = self.predict_logit(x)
        loss = None
    
        if targets != None:
            targets = targets.to(self.device)
            loss = self.calcul_loss_pheno(logits, targets, loss_version=self.loss_version)
        
        return logits, loss         
    
    def predict_pheno(self, pheno_sentence):
        self.eval()
        logits, loss = self.forward(pheno_sentence)
        return torch.argmax(logits, axis=1)

    def calcul_loss_pheno(self, logits, targets=None, loss_version='cross_entropy'):
        if targets is None:
            loss = None
        else:
            logits_similarities_embed = self.Embedding.similarities_tab[targets-1] #-1 to get at the level of without padding
            loss = F.cross_entropy(logits, logits_similarities_embed )

        return loss

    def evaluate(self, dataloader_test):
        print('beginning inference evaluation')
        start_time_inference = time.time()
        predicted_labels_list = []
        predicted_probas_list = []
        true_labels_list = []

        total_loss = 0.
        self.eval()
        with torch.no_grad():
            for batch_sentences_pheno, batch_labels_pheno in dataloader_test:


                logits, loss = self(batch_sentences_pheno, targets=batch_labels_pheno)
                total_loss += loss.item()
                predicted_labels = self.predict_pheno(batch_sentences_pheno)
                predicted_labels_list.extend(predicted_labels.cpu().numpy())
                predicted_probas = F.softmax(logits, dim=1)
                predicted_probas_list.extend(predicted_probas.cpu().numpy())
                true_labels_list.extend(batch_labels_pheno.cpu().numpy())
        f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')
        accuracy = accuracy_score(true_labels_list, predicted_labels_list)
        auc_score = 0#calculate_roc_auc(true_labels_list, np.array(predicted_probas_list)[:, 1], return_nan=True)
        proba_avg_zero, proba_avg_one = get_proba(true_labels_list, predicted_probas_list)
    
        self.train()
        print(f'end inference evaluation in {time.time() - start_time_inference}s')
        return f1, accuracy, auc_score, total_loss/len(dataloader_test), proba_avg_zero, proba_avg_one, predicted_probas_list, true_labels_list


        
            
        
            
   

class PadMaskSequential(nn.Sequential):
    def __init__(self, *args):
        super(PadMaskSequential, self).__init__(*args)
        self.padding_mask = None

    def set_padding_mask_sequential(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, x, padding_mask):
        for module in self:
           # module.set_padding_mask_block(self.padding_mask)
            x = module(x, padding_mask)
        return x
   
class BlockPheno(nn.Module):
    def __init__(self, instance_size, n_head, Head_size, p_dropout):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_head, Head_size, instance_size, p_dropout)
        self.ffwd = FeedForward(instance_size, p_dropout)
        self.ln1 = nn.LayerNorm(instance_size)
        self.ln2 = nn.LayerNorm(instance_size)
        self.padding_mask = None

    def set_padding_mask_block(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, x, padding_mask=None):
        #self.sa.set_padding_mask_sa(self.padding_mask)
        x = x + self.sa(self.ln1(x), padding_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
    
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, Head_size, instance_size, p_dropout):
        super().__init__()
        self.qkv_network = nn.Linear(instance_size, Head_size * 3, bias = False) #group the computing of the nn.Linear for q, k and v, shape 
        self.proj = nn.Linear(Head_size, instance_size)
        self.attention_dropout = nn.Dropout(p_dropout)
        self.resid_dropout = nn.Dropout(p_dropout)

        self.multi_head_size = Head_size // n_head
        self.flash = False
        self.n_head = n_head
        self.Head_size = Head_size
        self.padding_mask = None

    def set_padding_mask_sa(self, padding_mask):
        self.padding_mask = padding_mask

        #self.dropout = nn.Dropout(dropout)
    def forward(self, x, padding_mask=None):
        # x of size (B, S, E)
        Batch_len, Sentence_len, _ = x.shape
        q, k, v  = self.qkv_network(x).split(self.Head_size, dim=2) #q, k, v of shape (B, S, H)
        # add a dimension to compute the different attention heads separately
        q_multi_head = q.view(Batch_len, Sentence_len, self.n_head, self.multi_head_size).transpose(1, 2) #shape B, HN, S, MH
        k_multi_head = k.view(Batch_len, Sentence_len, self.n_head, self.multi_head_size).transpose(1, 2)
        v_multi_head = v.view(Batch_len, Sentence_len, self.n_head, self.multi_head_size).transpose(1, 2)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q_multi_head, k_multi_head, v_multi_head, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:    
            attention_weights = (q_multi_head @ k_multi_head.transpose(-2, -1))/np.sqrt(self.multi_head_size) # shape B, S, S
            ### padding mask #####
            attention_probas = F.softmax(attention_weights, dim=-1) # shape B, S, S
            if padding_mask != None:
                attention_probas = (attention_probas.transpose(0, 1)*padding_mask).transpose(0, 1)
           # attention_probas[attention_probas.isnan()]=0
            attention_probas = self.attention_dropout(attention_probas)


            #print(f'wei1={attention_probas}')
            #attention_probas = self.dropout(attention_probas)
            ## weighted aggregation of the values
            out = attention_probas @ v_multi_head # shape B, S, S @ B, S, MH = B, S, MH
        out = out.transpose(1, 2).contiguous().view(Batch_len, Sentence_len, self.Head_size)
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out        
    
  
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""
    def __init__(self, instance_size, p_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( instance_size, 4 * instance_size),
            nn.ReLU(),
            nn.Linear(4 * instance_size, instance_size),
            nn.Dropout(p_dropout)
        )

    def forward(self, x):
        return self.net(x)
 
    
        
    
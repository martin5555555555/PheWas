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


class PhenotypeEncoding(nn.Module):
    def __init__(self, Embedding, Head_size, n_head, n_layer, mask_padding=False, padding_token=None, p_dropout=0, device='cpu', instance_size=None, proj_embed=True):
        super().__init__()
       
        self.Embedding_size = Embedding.Embedding_size
        self.instance_size=instance_size
        self.proj_embed = proj_embed
        if not self.proj_embed:
            self.instance_size = self.Embedding_size
        if self.proj_embed:
            self.projection_embed = nn.Linear(self.Embedding_size, self.instance_size)
        self.blocks =PadMaskSequential(*[BlockPheno(self.instance_size, n_head=n_head, Head_size=Head_size, p_dropout=p_dropout) for _ in range(n_layer)]) #Block(self.instance_size, n_head=n_head, Head_size=Head_size) 
        self.ln_f = nn.LayerNorm(self.instance_size) # final layer norm
        self.Embedding = Embedding
        self.mask_padding = mask_padding
        self.padding_token = padding_token
        self.padding_mask = None
        self.device = device
       
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
        padding_mask_mat = torch.ones((B, S, S), dtype=torch.int)
        padding_mask_mat[mask] = 0
        padding_mask_mat.transpose(-2,-1)[mask] = 0

        
        padding_mask_probas = torch.zeros((B, S))
        padding_mask_probas[mask] = -torch.inf
        padding_mask_probas = padding_mask_probas.view(B, S)
        return padding_mask_mat.to(self.device), padding_mask_probas.to(self.device) # 1 if masked, 0 else

    def set_padding_mask_transformer(self, padding_mask, padding_mask_probas):
        self.padding_mask = padding_mask
        self.padding_mask_probas = padding_mask_probas
    
    def forward(self, diseases_sentence):
        Batch_len, Sentence_len = diseases_sentence.shape

        diseases_sentence = diseases_sentence.to(self.device)
        #counts_diseases = counts_diseases.to(self.device)
        
        if self.mask_padding:
            padding_mask, padding_mask_probas = self.create_padding_mask(diseases_sentence)
            self.set_padding_mask_transformer(padding_mask, padding_mask_probas)
            self.blocks.set_padding_mask_sequential(self.padding_mask)

        diseases_sentences_embedded = self.diseases_embedding_table(diseases_sentence) # shape B, S, E

        x = diseases_sentences_embedded 
    
        #if self.pheno_method == 'Paul':
        #    counts_diseases_embedded = self.counts_embedding_table(counts_diseases) # shape B, S, E
        #    #x = x + counts_diseases_embedded # shape B, S, E 
        
        if self.proj_embed:
            x = self.projection_embed(x)
        x = self.blocks(x) # shape B, S, E
        
        return x
   

class PadMaskSequential(nn.Sequential):
    def __init__(self, *args):
        super(PadMaskSequential, self).__init__(*args)
        self.padding_mask = None

    def set_padding_mask_sequential(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, x):
        for module in self:
            module.set_padding_mask_block(self.padding_mask)
            x = module(x)
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

    def forward(self, x):
        self.sa.set_padding_mask_sa(self.padding_mask)
        x = x + self.sa(self.ln1(x))
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
    def forward(self, x):
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
            if self.padding_mask != None:
                padding_mask_weights = -(1-self.padding_mask)*(10**10)
                attention_weights = (attention_weights.transpose(0, 1)+padding_mask_weights).transpose(0, 1)
            #print(f'wei0={attention_weights}')
            attention_probas = F.softmax(attention_weights, dim=-1) # shape B, S, S
            if self.padding_mask != None:
                attention_probas = (attention_probas.transpose(0, 1)*self.padding_mask).transpose(0, 1)
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
 

class SNPEncoding(nn.Module):
    def __init__(self, Embedding, Head_size, n_head, n_layer, p_dropout=0, device='cpu'):
        super().__init__()
       
        self.Embedding_size = Embedding.Embedding_size
        self.instance_size = self.Embedding_size
        self.blocks = nn.Sequential(*[BlockSNP(self.instance_size, n_head=n_head, Head_size=Head_size, p_dropout=p_dropout) for _ in range(n_layer)]) #Block(self.instance_size, n_head=n_head, Head_size=Head_size) 
        self.ln_f = nn.LayerNorm(self.instance_size) # final layer norm
        self.Embedding = Embedding
        self.device = device
       
        self.SNPS_embedding_table = Embedding.SNPS_embeddings
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
        padding_mask_mat = torch.zeros((B, S, S), dtype=torch.bool)
        padding_mask_mat[mask] = 1
        padding_mask_mat.transpose(-2,-1)[mask] = 1

        padding_mask_probas = torch.zeros((B, S)).to(bool)
        padding_mask_probas[mask] = True
        padding_mask_probas = padding_mask_probas.view(B, S)
        return padding_mask_mat, padding_mask_probas # 1 if masked, 0 else

    def set_padding_mask_transformer(self, padding_mask, padding_mask_probas):
        self.padding_mask = padding_mask
        self.padding_mask_probas = padding_mask_probas
    
    def forward(self, SNPS_sentence):
        Batch_len, Nb_SNP = SNPS_sentence.shape
        pos_SNPS = torch.arange(Nb_SNP)*2
        SNPS_sentence = SNPS_sentence #+ pos_SNPS # Shape B, nb_SNPS*2
        SNP_sentences_embedded = self.SNPS_embedding_table(SNPS_sentence) # shape B, Nb_SNP, E

        #if self.pheno_method == 'Paul':
        #    counts_diseases_embedded = self.counts_embedding_table(counts_diseases) # shape B, S, E
        #    #x = x + counts_diseases_embedded # shape B, S, E 
        x = self.blocks(SNP_sentences_embedded) # shape B, S, E
        
        return x
   

class BlockSNP(nn.Module):
    def __init__(self, instance_size, n_head, Head_size, p_dropout):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_head, Head_size, instance_size, p_dropout)
        self.ffwd = FeedForward(instance_size, p_dropout)
        self.ln1 = nn.LayerNorm(instance_size)
        self.ln2 = nn.LayerNorm(instance_size)
        self.padding_mask = None

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CrossMultiAttentionSNPPheno(nn.Module):
        # Key are the phenos, Queries are the SNPS
    def __init__(self, n_head, Head_size, instance_size_pheno, instance_size_SNPS, p_dropout):
        super().__init__()
        self.queries_network = nn.Linear(instance_size_pheno, Head_size, bias=False)
        self.keys_network = nn.Linear(instance_size_pheno, Head_size, bias=False)
        self.values_network_SNP = nn.Linear(instance_size_SNPS, Head_size, bias=False)
        self.values_network_pheno = nn.Linear(instance_size_pheno, Head_size, bias=False)


        self.attention_dropout = nn.Dropout(p_dropout)
        self.resid_dropout = nn.Dropout(p_dropout)

        self.multi_head_size = Head_size // n_head
        self.n_head = n_head
        self.Head_size = Head_size
        self.padding_mask = None

    def set_padding_mask_sa(self, padding_mask):
        self.padding_mask = padding_mask

        #self.dropout = nn.Dropout(dropout)
    def forward(self, pheno_encoded, SNPS_encoded):
        # x of size (B, S, E)
        Batch_len, Sentence_len_pheno, _ = pheno_encoded.shape
        Batch_len, Sentence_len_SNPS, _ = SNPS_encoded.shape
        keys = self.keys_network(pheno_encoded)
        queries = self.queries_network(SNPS_encoded) 
             
        values_pheno = self.values_network_pheno(pheno_encoded)
        values_SNPS = self.values_network_SNP(SNPS_encoded)
       
    
        # add a dimension to compute the different attention heads separately
        q_multi_head = queries.view(Batch_len, Sentence_len_SNPS, self.n_head, self.multi_head_size).transpose(1, 2) #shape B, HN, S_SNPS, MH
        k_multi_head = keys.view(Batch_len, Sentence_len_pheno, self.n_head, self.multi_head_size).transpose(1, 2)#shape B, HN, S_PHENO, MH
        values_pheno_multihead = values_pheno.view(Batch_len, Sentence_len_pheno, self.n_head, self.multi_head_size).transpose(1, 2)
        values_SNPS_multihead = values_SNPS.view(Batch_len, Sentence_len_SNPS, self.n_head, self.multi_head_size).transpose(1, 2)
        attention_weights = (k_multi_head @ q_multi_head.transpose(-2, -1))/np.sqrt(self.multi_head_size) # shape B, HN, S_PHENO, S_SNPS
        ### padding mask #####
        if self.padding_mask != None:
            attention_weights[self.padding_mask] = 1**-10     #float('-inf')
        #print(f'wei0={attention_weights}')
        

        #print(f'wei1={attention_probas}')
        #attention_probas = self.dropout(attention_probas)
        ## weighted aggregation of the values
       
        attention_probas_phenos = F.softmax(attention_weights, dim=-1) # shape B, S, S
        attention_probas_SNPS = F.softmax(attention_weights.transpose(-1, -2), dim=-1) # shape B, S, S


        attention_probas_phenos = self.attention_dropout(attention_probas_phenos)
        attention_probas_SNPS = self.attention_dropout(attention_probas_SNPS)

        out_pheno = attention_probas_phenos @ values_SNPS_multihead  # shape B, S, S @ B, S, MH = B, S, MH
        out_SNPS = attention_probas_SNPS @ values_pheno_multihead
        
        out_pheno = out_pheno.transpose(1, 2).contiguous().view(Batch_len, Sentence_len_pheno, self.Head_size) + values_pheno
        out_SNPS = out_SNPS.transpose(1, 2).contiguous().view(Batch_len, Sentence_len_SNPS, self.Head_size) + values_SNPS

        out_pheno = self.resid_dropout(out_pheno)
        out_SNPS = self.resid_dropout(out_SNPS)
        return out_pheno, out_SNPS       
    
class BlockCrossSNPPHENO(nn.Module):
    def __init__(self, instance_size_SNPS, instance_size_pheno, n_head, Head_size, p_dropout):
        super().__init__()
        self.ca = CrossMultiAttentionSNPPheno(n_head=n_head, Head_size=Head_size, instance_size_SNPS=instance_size_SNPS, 
                                             instance_size_pheno=instance_size_pheno,
                                              p_dropout=p_dropout)       
        self.ffwd_pheno = FeedForward(instance_size_pheno, p_dropout)
        self.ffwd_SNPS = FeedForward(instance_size_SNPS, p_dropout)

        self.ln1_pheno = nn.LayerNorm(instance_size_pheno)
        self.ln1_SNPS = nn.LayerNorm(instance_size_SNPS)

        self.proj_pheno = nn.Linear(Head_size, instance_size_pheno)
        self.proj_SNPS = nn.Linear(Head_size, instance_size_SNPS)



        self.padding_mask = None

    def forward(self, encoded_phenos, encoded_SNPS):
        encoded_phenos = self.ln1_pheno(encoded_phenos)
        encoded_SNPS = self.ln1_SNPS(encoded_SNPS)

        out_pheno, out_SNPS = self.ca(encoded_phenos, encoded_SNPS)

        
        out_pheno = self.proj_pheno(out_pheno)
        out_pheno = self.ln1_pheno(out_pheno)
        out_pheno = out_pheno + self.ffwd_pheno(out_pheno)

        out_SNPS = self.proj_SNPS(out_SNPS)
        out_SNPS = self.ln1_SNPS(out_SNPS) 
        out_SNPS = out_SNPS + self.ffwd_SNPS(out_SNPS)  
            
        return out_pheno, out_SNPS
    
class CrossPadMaskSequential(nn.Sequential):
    def __init__(self, *args):
        super(CrossPadMaskSequential, self).__init__(*args)
        self.padding_mask = None

    def set_padding_mask_sequential(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, encoded_phenos, encoded_SNPS):
        for module in self:
            encoded_phenos, encoded_SNPS = module(encoded_phenos, encoded_SNPS)
        return encoded_phenos, encoded_SNPS
   
class PredictLogit(nn.Module):
    def __init__(self, instance_size_SNPS, instance_size_pheno, nb_phenos_possible):
        super().__init__()
        self.ln2_phenos = nn.LayerNorm(instance_size_pheno)
        self.ln2_SNPS = nn.LayerNorm(instance_size_SNPS)

        
        self.get_logits_phenos = nn.Linear(instance_size_pheno, nb_phenos_possible)
        self.get_logits_SNPS = nn.Linear(instance_size_SNPS, 2)
    
    def forward(self, pheno_sentence, SNPS_sentence, value):
        if value == 'pheno':
            pheno_sentence = self.ln2_phenos(pheno_sentence)
            logits = self.get_logits_phenos(pheno_sentence)
        else:
            SNPS_sentence = self.ln2_SNPS(SNPS_sentence)
            logits = self.get_logits_SNPS(SNPS_sentence)
        logits_mean = logits.mean(axis=1)
        return logits_mean

class GenerativeModelPheWasV1(nn.Module):
    def __init__(self, n_head_pheno, Head_size_pheno, Embedding_pheno, Embedding_SNPS, instance_size_pheno,
                n_layer_pheno,  nb_SNPS, n_layer_SNPS, n_head_SNPS, Head_size_SNPS, instance_size_SNPS, nb_phenos_possible,
                n_head_cross, Head_size_cross, n_layer_cross, p_dropout, device='cpu', mask_padding=True,
                loss_version_pheno='cross_entropy', loss_version_SNPS='cross_entropy', gamma=2, alpha=1, padding_token=0):
        super().__init__()
        print(device, flush=True)
        self.Embedding_pheno = Embedding_pheno
        self.Embedding_SNPS = Embedding_SNPS
        self.Embedding_size_pheno = Embedding_pheno.Embedding_size
        self.Embedding_size_SNP = Embedding_SNPS.Embedding_size
        
        self.instance_size_pheno = instance_size_pheno
        self.n_head_pheno = n_head_pheno
        self.Head_size_pheno = Head_size_pheno
        self.n_layer_pheno = n_layer_pheno
        self.loss_version_pheno = loss_version_pheno

        self.Head_size_SNPS = Head_size_SNPS
        self.n_layer_SNPS = n_layer_SNPS
        self.nb_SNPS = nb_SNPS
        self.n_head_SNPS = n_head_SNPS
        self.instance_size_SNPS = instance_size_SNPS
        self.loss_version_SNPS = loss_version_SNPS


        self.gamma = gamma
        self.alpha = alpha

        self.n_layer_cross = n_layer_cross
        self.Head_size_cross = Head_size_cross
        self.n_head_cross = n_head_cross
        self.p_dropout = p_dropout

       


        self.nb_phenos_possible = nb_phenos_possible
        self.device = device
        self.padding_token = padding_token

      

        self.encoding_phenos = PhenotypeEncoding(Embedding=Embedding_pheno, Head_size=Head_size_pheno, 
            n_head=n_head_pheno, n_layer=n_layer_pheno, instance_size=instance_size_pheno, device=device, mask_padding=mask_padding,
            p_dropout=p_dropout, padding_token=self.padding_token)
        self.encoding_SNPS = SNPEncoding(Embedding=Embedding_SNPS, Head_size=Head_size_SNPS, n_head=n_head_SNPS,
                    device=device, n_layer=n_layer_pheno, p_dropout=p_dropout)
        self.blocks = CrossPadMaskSequential(*[ BlockCrossSNPPHENO(n_head=n_head_cross, Head_size=Head_size_cross, 
                                             instance_size_SNPS=instance_size_SNPS, 
                                             instance_size_pheno=instance_size_pheno,
                                             p_dropout=p_dropout) for _ in range(n_layer_cross)]) #Block(self.instance_size, n_head=n_head, Head_size=Head_size) 

        self.predict_logit = PredictLogit(instance_size_pheno=instance_size_pheno, instance_size_SNPS=instance_size_SNPS, nb_phenos_possible=nb_phenos_possible-1) #-1 for padding



    def forward(self, diseases_sentence, SNPS_sentence, value, targets=None):
        diseases_sentence = diseases_sentence.to(self.device)
        SNPS_sentence = SNPS_sentence.to(self.device)
        
        phenotype_encoded = self.encoding_phenos(diseases_sentence)
        SNPS_encoded = self.encoding_SNPS(SNPS_sentence)

        out_pheno, out_SNPS = self.blocks(phenotype_encoded, SNPS_encoded)
        logits = self.predict_logit(out_pheno, out_SNPS, value)

        loss = None
    
        if targets != None:
            targets = targets.to(self.device)
            if value == 'pheno':
                loss = self.calcul_loss_pheno(logits, targets, loss_version=self.loss_version_pheno)
            elif value == 'SNP':
                loss  = self.calcul_loss_SNPS(logits, targets, loss_version=self.loss_version_SNPS, gamma=self.gamma, alpha=self.alpha)
        
        return logits, loss

    def calcul_loss_SNPS(self, logits, targets=None, loss_version='cross_entropy', gamma=None, alpha=None):
        if targets is None:
            loss = None
        else:
            #target : shape B,
            
            if loss_version == 'cross_entropy':
                cross_entropy = F.cross_entropy(logits, targets)
                return cross_entropy
            elif loss_version == 'focal_loss':
                alphas = ((1 - targets) * (alpha-1)).to(torch.float) + 1
                probas = F.softmax(logits)
                certidude_factor =  (1-probas[[range(probas.shape[0]), targets]])**gamma * alphas
                cross_entropy = F.cross_entropy(logits, targets, reduce=False)
                loss = torch.dot(cross_entropy, certidude_factor)
                return loss
            elif loss_version == 'predictions':
                probas = F.softmax(logits)
                predictions = (probas[:,1] > 0.5).to(int)
                return torch.sum((predictions-targets)**2)/len(predictions)
        
    def predict_pheno(self, pheno_sentence, SNPS_sentences):
        self.eval()
        logits, loss = self.forward(pheno_sentence, SNPS_sentences, value='pheno')
        return torch.argmax(logits, axis=1)
        self.train()

    def calcul_loss_pheno(self, logits, targets=None, loss_version='cross_entropy'):
        if targets is None:
            loss = None
        else:
            logits_similarities_embed = self.Embedding_pheno.similarities_tab[targets-1] #-1 to get at the level of without padding
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
            for batch_sentences_pheno, batch_labels_pheno, batch_sentences_SNPS in dataloader_test:


                logits, loss = self(batch_sentences_pheno, batch_sentences_SNPS,  value = 'pheno', targets=batch_labels_pheno)
                total_loss += loss.item()
                predicted_labels = self.predict_pheno(batch_sentences_pheno, batch_sentences_SNPS)
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


        
            
        
            
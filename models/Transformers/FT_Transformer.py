##### Functionnal version with optionnal mask padding and dropouts, see Transformer_V1.ipynb for example
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)


### imports
import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from codes.models.data_form.DataForm import DataTransfo_1SNP, PatientList, Patient

from codes.models.Transformers.Embedding import EmbeddingPhenoCat
from codes.models.metrics import calculate_roc_auc, calculate_classification_report, calculate_loss, get_proba
from torch.utils.data import DataLoader
### Transformer's instance
# B, S, E, H, HN, MH = Batch_len, Sentence_len, Embedding_len, Head_size, Head number, MultiHead size.
class TabTransformerGeneModel_V2(nn.Module):
    def __init__(self, pheno_method, Embedding, instance_size, proj_embed, list_env_features, Head_size, binary_classes, n_head, n_layer, mask_padding=False, padding_token=None, p_dropout=0, device='cpu', loss_version='cross_entropy', gamma=2, alpha=1):
        super().__init__()
       
        self.Embedding_size = Embedding.Embedding_size
        

        self.mask_padding = mask_padding
        self.padding_token = padding_token
        self.padding_mask = None
        self.device = device
        self.pheno_method = pheno_method
        self.binary_classes = binary_classes
        self.Classes_nb = 2 if self.binary_classes else 3
        self.loss_version = loss_version
        self.gamma = gamma
        self.alpha = alpha
        self.list_env_features = list_env_features
        self.nb_env = len(self.list_env_features)
        self.instance_size = instance_size
        self.Embedding = Embedding
        self.proj_embed = proj_embed
        if not self.proj_embed:
            self.instance_size = self.Embedding_size
        if self.proj_embed:
            self.projection_embed = nn.Linear(self.Embedding_size, self.instance_size)
        
        self.blocks =PadMaskSequential(*[Block(self.instance_size, n_head=n_head, Head_size=Head_size, p_dropout=p_dropout, nb_env=self.nb_env) for _ in range(n_layer)]) #Block(self.instance_size, n_head=n_head, Head_size=Head_size) 
        self.ln_f = nn.LayerNorm(self.instance_size) # final layer norm
        self.lm_head_logits = nn.Linear(self.instance_size, self.Classes_nb) 
        self.lm_head_proba = nn.Linear(self.instance_size,1) # plus one for the probabilities


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            

    def create_padding_mask(self, input_dict):
        diseases_sentence = input_dict['diseases']
        B, S = np.shape(diseases_sentence)[0], np.shape(diseases_sentence)[1]
        mask = torch.where(diseases_sentence==self.padding_token)
        padding_mask_mat = torch.ones((B, S+self.nb_env, S+self.nb_env))
        padding_mask_mat[mask] = 0
        padding_mask_mat.transpose(-2,-1)[mask] = 0

        padding_mask_probas = torch.ones((B, S+self.nb_env))
        padding_mask_probas[mask] = 0
        padding_mask_probas = padding_mask_probas.view(B, S+self.nb_env)
        return padding_mask_mat.to(self.device), padding_mask_probas.to(self.device) # 1 if masked, 0 else

    def set_padding_mask_transformer(self, padding_mask_mat, padding_mask_probas):
        self.padding_mask_mat = padding_mask_mat
        self.padding_mask_probas = padding_mask_probas
    

    def forward(self, input_dict):
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)

        if 'SNP_label' in list(input_dict.keys()):
            targets = input_dict.pop('SNP_label')
        else:
            targets = None
        input_embedded = self.Embedding(input_dict)
        Batch_len, Sentence_len, _ = input_embedded.shape

   

        if self.mask_padding:
            padding_mask_mat, padding_mask_probas = self.create_padding_mask(input_dict)
            self.set_padding_mask_transformer(padding_mask_mat, padding_mask_probas)
            self.blocks.set_padding_mask_sequential(self.padding_mask_mat)

        
        x = self.blocks(input_embedded) # shape B, S, E
        x = self.ln_f(x) # shape B, S, E
        logits = self.lm_head_logits(x) #shape B, S, Classes_Numb, token logits
        weights_logits = self.lm_head_proba(x).view(Batch_len, Sentence_len)
        probas = F.softmax(weights_logits) # shape B, S(represent the probas to be chosen)
        probas = probas * self.padding_mask_probas
        logits = (logits.transpose(1, 2) @ probas.view(Batch_len, Sentence_len, 1)).view(Batch_len, self.Classes_nb)# (B,Classes_Numb) Weighted Average logits
        loss = calculate_loss(logits, targets, self.loss_version, self.gamma, self.alpha)
        return logits, loss
    
    
    def forward_decomposed(self, input_dict):
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)
            
        if 'SNP_label' in list(input_dict.keys()):
            targets = input_dict.pop('SNP_label')
        else:
            targets = None
        input_embedded = self.Embedding(input_dict)
        Batch_len, Sentence_len, _ = input_embedded.shape


        if self.mask_padding:
            padding_mask_mat, padding_mask_probas = self.create_padding_mask(input_dict)
            self.set_padding_mask_transformer(padding_mask_mat, padding_mask_probas)
            self.blocks.set_padding_mask_sequential(self.padding_mask_mat)

        
        x, attention_probas = self.blocks.forward_decompose(input_embedded) # shape B, S, E
        x = self.ln_f(x) # shape B, S, E
        logits = self.lm_head_logits(x) #shape B, S, Classes_Numb, token logits
        weights_logits = self.lm_head_proba(x).view(Batch_len, Sentence_len)
        probas = F.softmax(weights_logits) # shape B, S(represent the probas to be chosen)
        probas = probas * self.padding_mask_probas

        logits = (logits.transpose(1, 2) @ probas.view(Batch_len, Sentence_len, 1)).view(Batch_len, self.Classes_nb)# (B,Classes_Numb) Weighted Average logits
        loss = calculate_loss(logits, targets, self.loss_version, self.gamma, self.alpha)
        return logits, loss, input_embedded, attention_probas, probas

    def predict(self,input_dict):
        if 'SNP_label' in list(input_dict.keys()):
            input_dict.pop('SNP_label')
        logits, _ = self(input_dict) # shape B, Classes_nb
        return torch.argmax(logits, dim=1)  # (B,)
        
    def predict_proba(self, input_dict):
        if 'SNP_label' in list(input_dict.keys()):
            input_dict.pop('SNP_label')
        logits, _ = self(input_dict)
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def evaluate(self, dataloader_test):
        print('beginning inference evaluation')
        start_time_inference = time.time()
        predicted_labels_list = []
        predicted_probas_list = []
        true_labels_list = []

        total_loss = 0.
        self.eval()
        with torch.no_grad():
            for input_dicts in dataloader_test:

                batch_labels = input_dicts['SNP_label']

                logits, loss = self(input_dicts)
                total_loss += loss.item()
                predicted_labels = self.predict(input_dicts)
                predicted_labels_list.extend(predicted_labels.cpu().numpy())
                predicted_probas = self.predict_proba(input_dicts)
                predicted_probas_list.extend(predicted_probas.cpu().numpy())
                true_labels_list.extend(batch_labels.cpu().numpy())
        f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')
        accuracy = accuracy_score(true_labels_list, predicted_labels_list)
        auc_score = calculate_roc_auc(true_labels_list, np.array(predicted_probas_list)[:, 1], return_nan=True)
        proba_avg_zero, proba_avg_one = get_proba(true_labels_list, predicted_probas_list)
    
        self.train()
        print(f'end inference evaluation in {time.time() - start_time_inference}s')
        return f1, accuracy, auc_score, total_loss/len(dataloader_test), proba_avg_zero, proba_avg_one, predicted_probas_list, true_labels_list


    def write_embedding(self, writer):
        if self.proj_embed:
            embedding_tensor = self.projection_embed(self.Embedding.dic_embedding_cat['diseases'].weight).detach().cpu().numpy()
        else:
            embedding_tensor = self.Embedding.dic_embedding_cat['diseases'].weight.detach().cpu().numpy()
        writer.add_embedding(embedding_tensor, metadata=self.Embedding.metadata, metadata_header=["Name","Label"])


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
    
    def forward_decompose(self, x):
        for module in self:
            module.set_padding_mask_block(self.padding_mask)
            x = module.forward_decompose(x)
        return x
    
class Block(nn.Module):
    def __init__(self, instance_size, n_head, Head_size, p_dropout, nb_env):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_head, Head_size, instance_size, p_dropout,  nb_env=nb_env)
        self.ffwd = FeedForward(instance_size, p_dropout)
        self.ln1 = nn.LayerNorm(instance_size)
        self.ln2 = nn.LayerNorm(instance_size)
        self.padding_mask = None

    def set_padding_mask_block(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, x):
        self.sa.set_padding_mask_sa(self.padding_mask)
        #x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x
    
    def forward_decompose(self, x):
        self.sa.set_padding_mask_sa(self.padding_mask)
        out_sa, attention_probas= self.sa.forward_decompose(x)
        x = out_sa + x
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x, attention_probas


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, Head_size, instance_size, p_dropout, nb_env):
        super().__init__()
        self.q_network = nn.Linear(instance_size, Head_size, bias = False) 
        self.k_network =  nn.Linear(instance_size, Head_size, bias = False)
        self.v_network =  nn.Linear(instance_size, Head_size, bias = False)
        self.proj = nn.Linear(Head_size, instance_size)
        self.attention_dropout = nn.Dropout(p_dropout)
        self.resid_dropout = nn.Dropout(p_dropout)

        self.multi_head_size = Head_size // n_head
        self.flash = False
        self.n_head = n_head
        self.Head_size = Head_size
        self.padding_mask = None
        self.nb_env = nb_env

    def set_padding_mask_sa(self, padding_mask):
        self.padding_mask = padding_mask

        #self.dropout = nn.Dropout(dropout)
   
    def forward(self, x):
        # x of size (B, S, E)
        Batch_len, Sentence_len, _ = x.shape

        q = self.q_network(x)
        k = self.k_network(x)
        v = self.v_network(x)
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
            if self.padding_mask != None:
                attention_probas = (attention_probas.transpose(0, 1)*self.padding_mask).transpose(0, 1)
           # attention_probas[attention_probas.isnan()]=0
            attention_probas = self.attention_dropout(attention_probas)

            #print(f'wei1={attention_probas}')
            #attention_probas = self.dropout(attention_probas)
            ## weighted aggregation of the values
            out = attention_probas @ v_multi_head # shape B, S-Env, S @ B, S, MH = B, S, MH
        out = out.transpose(1, 2).contiguous().view(Batch_len, Sentence_len, self.Head_size)
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out        

    def forward_decompose(self, x):
        # x of size (B, S, E)
        Batch_len, Sentence_len, _ = x.shape

        q = self.q_network(x)
        k = self.k_network(x)
        v = self.v_network(x)
        # add a dimension to compute the different attention heads separately
        q_multi_head = q.view(Batch_len, Sentence_len, self.n_head, self.multi_head_size).transpose(1, 2) #shape B, HN, S, MH
        k_multi_head = k.view(Batch_len, Sentence_len, self.n_head, self.multi_head_size).transpose(1, 2)
        v_multi_head = v.view(Batch_len, Sentence_len, self.n_head, self.multi_head_size).transpose(1, 2)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q_multi_head, k_multi_head, v_multi_head, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:    
            ### padding mask #####
            attention_weights = (q_multi_head @ k_multi_head.transpose(-2, -1))/np.sqrt(self.multi_head_size) # shape B, S, S
            ### padding mask #####
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
        return out, attention_probas     

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
        

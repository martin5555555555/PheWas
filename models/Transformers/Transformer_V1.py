##### Functionnal version with optionnal mask padding and dropouts, see Transformer_V1.ipynb for example


#### imports
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score

from codes.models.Transformers.Embedding import EmbeddingPheno
from codes.models.metrics import calculate_roc_auc, calculate_classification_report, calculate_loss, get_proba

### Transformer's instance
# B, S, E, H, HN, MH = Batch_len, Sentence_len, Embedding_len, Head_size, Head number, MultiHead size.
class TransformerGeneModel_V1(nn.Module):
    def __init__(self, pheno_method, Embedding, Head_size, Classes_nb, n_head, n_layer, mask_padding=False, padding_token=None, p_dropout=0, device='cpu', loss_version='cross_entropy', gamma=2):
        super().__init__()
       
        self.Embedding_size = Embedding.Embedding_size
        self.blocks =PadMaskSequential(*[Block(self.Embedding_size, n_head=n_head, Head_size=Head_size, p_dropout=p_dropout) for _ in range(n_layer)]) #Block(self.Embedding_size, n_head=n_head, Head_size=Head_size) 
        self.ln_f = nn.LayerNorm(self.Embedding_size) # final layer norm
        self.lm_head = nn.Linear(self.Embedding_size, Classes_nb)
        self.mask_padding = mask_padding
        self.padding_token = padding_token
        self.padding_mask = None
        self.device = device
        self.pheno_method = pheno_method
        self.loss_version = loss_version
        self.gamma = gamma

        self.diseases_embedding_table = Embedding.distinct_diseases_embeddings
        if self.pheno_method == 'Paul':
            self.counts_embedding_table = Embedding.counts_embeddings
        

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

        return padding_mask_mat # 1 if masked, 0 else

    def set_padding_mask_transformer(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, diseases_sentence, counts_diseases, targets=None):
        

        diseases_sentence = diseases_sentence.to(self.device)
        counts_diseases = counts_diseases.to(self.device)
        if targets != None:
            targets = targets.to(self.device)
        
        if self.mask_padding:
            padding_mask = self.create_padding_mask(diseases_sentence)
            self.set_padding_mask_transformer(padding_mask)
            self.blocks.set_padding_mask_sequential(self.padding_mask)

       
        diseases_sentences_embedded = self.diseases_embedding_table(diseases_sentence) # shape B, S, E

        x = diseases_sentences_embedded 
        
        if self.pheno_method == 'Paul':
            counts_diseases_embedded = self.counts_embedding_table(counts_diseases) # shape B, S, E
            x = x + counts_diseases_embedded # shape B, S, E 
        
        x = self.blocks(x) # shape B, S, E
        x = self.ln_f(x) # shape B, S, E
        logits = self.lm_head(x) #shape B, S, Classes_Numb
        logits = torch.mean(logits, dim=1) # (B,Classes_Numb) Average tokens logits. Or should I do it before? Or max ? Possible to interpret logit as which word had the largest weight?

        loss = calculate_loss(logits, targets, self.loss_version, self.gamma)
        
        return logits, loss
    


    def predict(self, diseases_sentence, diseases_count):
        logits, _ = self(diseases_sentence, diseases_count) # shape B, Classes_nb
        return torch.argmax(logits, dim=1)  # (B,)
        
    def predict_proba(self, diseases_sentence, diseases_count):
        logits, _ = self(diseases_sentence, diseases_count)
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def evaluate(self, dataloader_test):
        predicted_labels_list = []
        predicted_probas_list = []
        true_labels_list = []

        total_loss = 0.
        self.eval()
        with torch.no_grad():
            for batch_sentences, batch_counts, batch_labels in dataloader_test:

                batch_sentences = batch_sentences.to(self.device)
                batch_counts = batch_counts.to(self.device)
                batch_labels = batch_labels.to(self.device)

        

                logits, loss = self(batch_sentences, batch_counts,  batch_labels)
                total_loss += loss.item()
                predicted_labels = self.predict(batch_sentences, batch_counts)
                predicted_labels_list.extend(predicted_labels.cpu().numpy())
                predicted_probas = self.predict_proba(batch_sentences, batch_counts)
                predicted_probas_list.extend(predicted_probas.cpu().numpy())
                true_labels_list.extend(batch_labels.cpu().numpy())
        f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')
        accuracy = accuracy_score(true_labels_list, predicted_labels_list)
        auc_score = calculate_roc_auc(true_labels_list, np.array(predicted_probas_list)[:, 1], return_nan=True)
        proba_avg_zero, proba_avg_one = get_proba(true_labels_list, predicted_probas_list)
    
        self.train()
        return f1, accuracy, auc_score, total_loss/len(dataloader_test), proba_avg_zero, proba_avg_one, predicted_probas_list, true_labels_list




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
    
class Block(nn.Module):
    def __init__(self, Embedding_size, n_head, Head_size, p_dropout):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_head, Head_size, Embedding_size, p_dropout)
        self.ffwd = FeedForward(Embedding_size, p_dropout)
        self.ln1 = nn.LayerNorm(Embedding_size)
        self.ln2 = nn.LayerNorm(Embedding_size)
        self.padding_mask = None

    def set_padding_mask_block(self, padding_mask):
        self.padding_mask = padding_mask

    def forward(self, x):
        self.sa.set_padding_mask_sa(self.padding_mask)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, Head_size, Embedding_size, p_dropout):
        super().__init__()
        self.qkv_network = nn.Linear(Embedding_size, Head_size * 3, bias = False) #group the computing of the nn.Linear for q, k and v, shape 
        self.proj = nn.Linear(Head_size, Embedding_size)
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
            attention_weights = (q @ k.transpose(-2, -1))/np.sqrt(self.multi_head_size) # shape B, S, S
            ### padding mask #####
            if self.padding_mask != None:
                attention_weights[self.padding_mask] = 1**-10     #float('-inf')
            #print(f'wei0={attention_weights}')
            attention_probas = F.softmax(attention_weights, dim=-1) # shape B, S, S
            attention_probas = self.attention_dropout(attention_probas)

            #print(f'wei1={attention_probas}')
            #attention_probas = self.dropout(attention_probas)
            ## weighted aggregation of the values
            out = attention_probas @ v # shape B, S, S @ B, S, MH = B, S, MH
        out = out.transpose(1, 2).contiguous().view(Batch_len, Sentence_len, self.Head_size)
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out        

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""
    def __init__(self, Embedding_size, p_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( Embedding_size, 4 * Embedding_size),
            nn.ReLU(),
            nn.Linear(4 * Embedding_size, Embedding_size),
            nn.Dropout(p_dropout)
        )

    def forward(self, x):
        return self.net(x)
        

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class EmbeddingPheno(nn.Module):
    def __init__(self, method=None, vocab_size=None, Embedding_size=None, rollup_depth=4, freeze_embed=False, dicts=None, device='cpu'):
        super(EmbeddingPheno, self).__init__()

        self.dicts = dicts
        self.rollup_depth = rollup_depth
        self.nb_distinct_diseases_patient = vocab_size
        self.Embedding_size = Embedding_size
        self.metadata = None
        self.device = device

        if self.dicts != None:
            id_dict = self.dicts['id']
            name_dict = self.dicts['name']
            cat_dict = self.dicts['cat']
            codes = list(id_dict.keys())
            diseases_present = self.dicts['diseases_present']
            self.metadata = [[name_dict[code], cat_dict[code]] for code in codes]

        
        if method == None:
            self.distinct_diseases_embeddings = nn.Embedding(vocab_size, Embedding_size)
            #self.counts_embeddings = nn.Embedding(max_count_same_disease, Embedding_size)
            torch.nn.init.normal_(self.distinct_diseases_embeddings.weight, mean=0.0, std=0.02)
           # torch.nn.init.normal_(self.counts_embeddings.weight, mean=0.0, std=0.02)

        elif method == 'Abby':
            embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Abby/embedding_abby_no_1_diseases.pth'
            pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
            self.Embedding_size = pretrained_weights_diseases.shape[1]

            self.distinct_diseases_embeddings = nn.Embedding.from_pretrained(pretrained_weights_diseases, freeze=freeze_embed)
            #self.counts_embeddings = nn.Embedding(max_count_same_disease, self.Embedding_size)



        elif method=='Paul':
            embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Paul_Glove/glove_UKBB_omop_rollup_closest_depth_{self.rollup_depth}_no_1_diseases.pth'
            pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
            self.Embedding_size = pretrained_weights_diseases.shape[1]

            self.distinct_diseases_embeddings = nn.Embedding.from_pretrained(pretrained_weights_diseases, freeze=freeze_embed)
            #self.counts_embeddings = nn.Embedding(max_count_same_disease, self.Embedding_size)

        embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Abby/embedding_abby_no_1_diseases.pth'
        pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
        pretrained_weights_diseases = pretrained_weights_diseases[1:]
        nb_phenos = pretrained_weights_diseases.shape[0]
        self.similarities_tab = torch.tensor(np.array([F.cosine_similarity(pretrained_weights_diseases, pretrained_weights_diseases[i], dim=-1).view(nb_phenos) for i in range(nb_phenos)])).to(self.device)
        

    def write_embedding(self, writer):
            embedding_tensor = self.distinct_diseases_embeddings.weight.data.detach().cpu().numpy()
            writer.add_embedding(embedding_tensor, metadata=self.metadata, metadata_header=["Name","Label"])
class EmbeddingSNPS(nn.Module):
    def __init__(self, method=None, nb_SNPS=1, Embedding_size=10, freeze_embed=False):
        super(EmbeddingSNPS, self).__init__()

        self.method = method
        self.Embedding_size = Embedding_size
        self.nb_SNPS = nb_SNPS

        if method == None:
            self.SNPS_embeddings = nn.Embedding(self.nb_SNPS*2, Embedding_size)
            #self.counts_embeddings = nn.Embedding(max_count_same_disease, Embedding_size)
            torch.nn.init.normal_(self.SNPS_embeddings.weight, mean=0.0, std=0.02)
           # torch.nn.init.normal_(self.counts_embeddings.weight, mean=0.0, std=0.02)
            

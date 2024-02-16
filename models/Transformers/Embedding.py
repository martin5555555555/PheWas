import os
import sys
sys.path.append('/gpfs/commons/groups/gursoy_lab/pmeddeb/phenotype_embedding')
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SineCosineEncoding(nn.Module):
    def __init__(self, Embedding_size, max_len=1000):
        super(SineCosineEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, Embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, Embedding_size, 2).float() * -(np.log(10000.0) / Embedding_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):

        return self.encoding.to(x.device)[x]

class ZeroEmbedding(nn.Module):
    def __init__(self, Embedding_size, max_len=1000):
        super(ZeroEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, Embedding_size)
       

    def forward(self, x):

        return self.encoding.to(x.device)[x]


class EmbeddingPheno(nn.Module):
    def __init__(self, method=None, counts_method=None, vocab_size=None, max_count_same_disease=None, Embedding_size=None, rollup_depth=4, freeze_embed=False, dicts=None):
        super(EmbeddingPheno, self).__init__()

        self.dicts = dicts
        self.rollup_depth = rollup_depth
        self.nb_distinct_diseases_patient = vocab_size
        self.Embedding_size = Embedding_size
        self.max_count_same_disease = None
        self.metadata = None
        self.counts_method = counts_method

        if self.dicts != None:
            id_dict = self.dicts['id']
            name_dict = self.dicts['name']
            cat_dict = self.dicts['cat']
            codes = list(id_dict.keys())
            diseases_present = self.dicts['diseases_present']
            self.metadata = [[name_dict[code], cat_dict[code]] for code in codes]

        
        if method == None:
            self.distinct_diseases_embeddings = nn.Embedding(vocab_size, Embedding_size)
            self.counts_embeddings = nn.Embedding(max_count_same_disease, Embedding_size)
            torch.nn.init.normal_(self.distinct_diseases_embeddings.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.counts_embeddings.weight, mean=0.0, std=0.02)

        elif method == 'Abby':
            embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Abby/embedding_abby_no_1_diseases.pth'
            pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
            self.Embedding_size = pretrained_weights_diseases.shape[1]

            self.distinct_diseases_embeddings = nn.Embedding.from_pretrained(pretrained_weights_diseases, freeze=freeze_embed)
            self.counts_embeddings = nn.Embedding(max_count_same_disease, self.Embedding_size)



        elif method=='Paul':
            embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Paul_Glove/glove_UKBB_omop_rollup_closest_depth_{self.rollup_depth}_no_1_diseases.pth'
            pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
            self.Embedding_size = pretrained_weights_diseases.shape[1]

            self.distinct_diseases_embeddings = nn.Embedding.from_pretrained(pretrained_weights_diseases, freeze=freeze_embed)
            if self.counts_method == 'SineCosine':
                self.counts_embeddings = SineCosineEncoding(self.Embedding_size, max_count_same_disease)
            elif self.counts_method == 'no_counts':
                self.counts_embeddings = ZeroEmbedding(self.Embedding_size, max_count_same_disease )
            else:

                self.counts_embeddings = nn.Embedding(max_count_same_disease, self.Embedding_size)
    def write_embedding(self, writer):
            embedding_tensor = self.distinct_diseases_embeddings.weight.data.detach().cpu().numpy()
            writer.add_embedding(embedding_tensor, metadata=self.metadata, metadata_header=["Name","Label"])


class EmbeddingPhenoCat(nn.Module):
    def __init__(self, pheno_method=None,  method=None, proj_embed=None, counts_method=None, Embedding_size=10, instance_size=10, rollup_depth=4, freeze_embed=False, dic_embedding_cat_params={}, dicts=None, device='cpu'):
        super(EmbeddingPhenoCat, self).__init__()

        self.rollup_depth = rollup_depth
        self.Embedding_size = Embedding_size
        self.max_count_same_disease = None
        self.dic_embedding_cat_params = dic_embedding_cat_params
        dic_embedding_cat = {}
        self.method = method
        self.pheno_method = pheno_method
        self.dicts = dicts
        self.proj_embed = proj_embed
        self.projection_embed = None
        self.instance_size = instance_size
        self.counts_method = counts_method

        self.device = device
        if self.dicts != None:
            id_dict = self.dicts['id']
            name_dict = self.dicts['name']
            cat_dict = self.dicts['cat']
            codes = list(id_dict.keys())
            diseases_present = self.dicts['diseases_present']
            self.metadata = [[name_dict[code], cat_dict[code]] for code in codes]

        for cat, max_number  in self.dic_embedding_cat_params.items():
        
            if cat=='diseases':
                if self.method == None:
                    dic_embedding_cat[cat] = nn.Embedding(max_number, Embedding_size)
                    torch.nn.init.normal_(dic_embedding_cat[cat].weight, mean=0.0, std=0.02)

                elif self.method == 'Abby':
                    embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Abby/embedding_abby_no_1_diseases.pth'
                    pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
                    self.Embedding_size = pretrained_weights_diseases.shape[1]
                    dic_embedding_cat[cat] = nn.Embedding.from_pretrained(pretrained_weights_diseases, freeze=freeze_embed).to(self.device)

            

                elif self.method=='Paul':
                    embedding_file_diseases = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Embeddings/Paul_Glove/glove_UKBB_omop_rollup_closest_depth_{self.rollup_depth}_no_1_diseases.pth'
                    pretrained_weights_diseases = torch.load(embedding_file_diseases)[diseases_present]
                    self.Embedding_size = pretrained_weights_diseases.shape[1]
                    dic_embedding_cat[cat] = nn.Embedding.from_pretrained(pretrained_weights_diseases, freeze=freeze_embed).to(self.device)
                    
            elif cat == 'counts':
                if self.pheno_method == 'Paul':
                    if self.counts_method[cat] == 'SineCosine':
                        dic_embedding_cat[cat] = SineCosineEncoding(self.instance_size, max_number).to(self.device)
                    elif self.counts_method[cat] == 'no_counts':
                        dic_embedding_cat[cat] = ZeroEmbedding(self.instance_size, max_number).to(self.device)
                    else:
                        dic_embedding_cat[cat] = nn.Embedding(max_number, self.instance_size).to(self.device)
                        torch.nn.init.normal_(dic_embedding_cat[cat].weight, mean=0.0, std=0.02)

            elif cat == 'age':
                if self.counts_method[cat] == 'SineCosine':
                    dic_embedding_cat[cat] = SineCosineEncoding(self.instance_size, max_number).to(self.device)
                elif self.counts_method[cat] == 'no_counts':
                    dic_embedding_cat[cat] = ZeroEmbedding(self.instance_size, max_number).to(self.device)
                else:
                    dic_embedding_cat[cat] = nn.Embedding(max_number, self.instance_size).to(self.device)
                    torch.nn.init.normal_(dic_embedding_cat[cat].weight, mean=0.0, std=0.02)

                    

            else:
                dic_embedding_cat[cat] = nn.Embedding(max_number, self.instance_size).to(self.device)
                torch.nn.init.normal_(dic_embedding_cat[cat].weight, mean=0.0, std=0.02)

        if self.proj_embed:
            self.projection_embed = nn.Linear(self.Embedding_size, self.instance_size).to(self.device)

        self.dic_embedding_cat = dic_embedding_cat

    def forward(self, input_dict):
        list_env_embedded = []
        for key, value in input_dict.items():
            
            batch_len = len(value)

            if key=='diseases':
                diseases_sentences_embedded = self.dic_embedding_cat[key](value)
                if self.proj_embed:
                    diseases_sentences_embedded = self.projection_embed(diseases_sentences_embedded)

            elif key=='counts':
                if self.pheno_method == 'Paul':
                    counts_sentence_embedded = self.dic_embedding_cat[key](value)
                    diseases_sentences_embedded = diseases_sentences_embedded + counts_sentence_embedded
            

            else:
                list_env_embedded.append(self.dic_embedding_cat[key](value).view(batch_len, 1, self.instance_size))

        env_embedded = torch.concat(list_env_embedded, dim=1)

        return torch.concat([diseases_sentences_embedded, env_embedded], dim=1)
            


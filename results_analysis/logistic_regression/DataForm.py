import numpy as np
import pandas as pd
import vcfpy
import pickle
from tqdm import tqdm
import os
import time

def get_paths_pheno_dicts(method, rollup_depth=None):
    if method == 'Paul':
        path_pheno_id_dict = f"/gpfs/commons/groups/gursoy_lab/pmeddeb/phenotype_embedding/vocab_dict/code2id_ukbb_omop_rolled_up_depth_{rollup_depth}_closest_ancestor.pickle"
        path_pheno_name_dict = f"/gpfs/commons/groups/gursoy_lab/pmeddeb/phenotype_embedding/vocab_dict/code2name_ukbb_omop_rolled_up_depth_{rollup_depth}_closest_ancestor.pickle"
        path_pheno_cat_dict = f"/gpfs/commons/groups/gursoy_lab/pmeddeb/phenotype_embedding/vocab_dict/code2cat_ukbb_omop_rolled_up_depth_{rollup_depth}_closest_ancestor.pickle"
    elif method == 'Abby':
        path_pheno_id_dict = '/gpfs/commons/groups/gursoy_lab/anewbury/embeddings/data/cohortId2ind.pickle'
        path_pheno_name_dict = '/gpfs/commons/groups/gursoy_lab/anewbury/embeddings/data/cohortId2name.pickle'
        path_pheno_cat_dict = '/gpfs/commons/groups/gursoy_lab/anewbury/embeddings/data/cohortId2name.pickle' #no separation between name and cat for Abby's method
    return path_pheno_id_dict, path_pheno_name_dict, path_pheno_cat_dict
def get_paths_pheno_file(method, rollup_depth=None):
        if method == 'Paul':
            pheno_file  = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Pheno/Paul/ukbb_omop_rolled_up_depth_{rollup_depth}_closest_ancestor.csv'
        elif method == 'Abby':
            pheno_file = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Pheno/Abby/phenotype_embedding_df_abby.csv'
        return pheno_file

class DataTransfo_1SNP:
    def __init__(self, SNP, CHR, method='Paul', rollup_depth=4, binary_classes=True, pad_token='<PAD>', padding=True, load_data=True, save_data=False, remove_none=True, compute_features=True, data_share=1, prop_train_test=0.8, equalize_label=False, seuil_diseases=None):
        self.SNP = SNP
        self.CHR = CHR
        self.rollup_depth = rollup_depth
        self.binary_classes = binary_classes
        self.pad_token = pad_token
        self.label_dict = None
        self.padding = padding
        self.path = "/gpfs/commons/datasets/controlled/ukbb-gursoylab/"
        self.load_data = load_data
        self.save_data = save_data
        self.remove_none = remove_none
        self.compute_features = compute_features
        self.data_share = data_share
        self.indices_train = None
        self.indices_test = None
        self.prop_train_test = prop_train_test
        self.method = method
        self.seuil_diseases = seuil_diseases
        self.equalize_label = equalize_label
        path_pheno_id_dict, path_pheno_name_dict, path_pheno_cat_dict = get_paths_pheno_dicts(method, rollup_depth)
        self.pheno_file = get_paths_pheno_file(method, rollup_depth)
        with open(path_pheno_id_dict, "rb") as f:
            self.pheno_id_dict = pickle.load(f)
        with open(path_pheno_name_dict, "rb") as f:
            self.name_dict = pickle.load(f)
        with open(path_pheno_cat_dict, "rb") as f:
            self.cat_dict = pickle.load(f)

        self.vocab_size = len(self.pheno_id_dict)

        self.pheno_id_dict[self.pad_token]= 0
        self.name_dict[self.pad_token]= self.pad_token
        self.cat_dict[self.pad_token]= self.pad_token

        self.eid_list = None
        


    def get_patientlist(self):
        if self.load_data:
            start_time = time.time()
            print(f"loading data")
            # Open the file in binary write mode
            data_file = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Training/SNPS/{str(self.CHR)}/{self.SNP}/{self.method}/PatientList_{self.SNP}.pkl'
            with open(data_file, 'rb') as file:
                patient_list = pickle.load(file)
            print(f"data_loaded in {time.time() - start_time} s")

            if  self.data_share != 1:
                patient_list.keep_share(self.data_share)
                self.indices_train = None
                self.indices_test = None
        else:
            print("building data")
            genectic_data = self.get_genetic_data()
            pheno_data_df = self.get_pheno_data()
            patient_list = PatientList(
                [self.get_eid_data(eid, pheno_data_df.get_group(eid), self.method) 
                for eid in tqdm(list(pheno_data_df.groups.keys())[:int(self.nb_eid * self.data_share)], desc="Processing", unit="group")], 
                self.pad_token, self.pheno_id_dict)
            
            print(len(patient_list))
        
        if self.remove_none:
            print("removing None values")
            patient_list.remove_none()
        if self.compute_features:
            print("computing features")
            patient_list.get_nb_distinct_diseases_tot()
            patient_list.get_nb_max_distinct_diseases_patient()
            patient_list.get_max_count_same_disease()
            self.get_indices_train_test(patient_list, self.prop_train_test)

        if self.padding:
            print("padding data")
            patient_list.padd_data()

        if self.seuil_diseases != None:
            patient_list.set_seuil_data(self.seuil_diseases)
            self.indices_train = None
            self.indices_test = None
        if self.equalize_label:
            patient_list.equalize_label()

        if self.save_data and self.data_share==1: # saves only if all data have been processed
            print("saving data")
            data_dir = f'/gpfs/commons/groups/gursoy_lab/mstoll/codes/Data_Files/Training/SNPS/{str(self.CHR)}/{self.SNP}/{self.method}/'
            data_file = os.path.join(data_dir, f'PatientList_{self.SNP}.pkl')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            with open(data_file, 'wb') as file:
                # Use pickle.dump() to serialize and save the object to the file
                pickle.dump(patient_list, file)

        return patient_list

    def get_pheno_data(self):
        start_time = time.time()
        if self.method == 'Paul':
            print('loading df Paul')
            df = pd.read_csv(self.pheno_file)
            df['concept_id'] = df['concept_id'].map(self.pheno_id_dict)
            self.eid_list = list(df['eid'].unique())
            self.nb_eid = len(self.eid_list)
            df_grouped = df.groupby('eid')
        elif self.method == 'Abby':
            print('loading df Abby')
            pheno_df = pd.read_csv(self.pheno_file)
            pheno_df.set_index(pheno_df.subject_id, inplace=True)
            pheno_df.drop('subject_id', axis=1, inplace=True)
            self.eid_list = list(pheno_df.index)
            self.nb_eid = len(self.eid_list)
            pheno_df.columns = pheno_df.columns.map(self.pheno_id_dict)
            df_grouped = pheno_df.groupby(pheno_df.index)
        print(f'df loaded in {time.time() - start_time} s for creating data')
        return df_grouped


    
    def get_eid_data(self, eid, df, method):
        if method == 'Paul':
            unique_codes = list(df['concept_id'].values)
            occurrences = list(df['condition_occurrence_count'].values)

            disease_sentence = [code for code in unique_codes]
            counts_sentence = [count for count in occurrences]
            #print(str(eid) in list(self.label_dict.keys()))
        elif method == 'Abby':
            df_t = df.transpose()
            df_t = df_t[df_t[eid]==1]
            disease_sentence = np.array(df_t.index)
            counts_sentence = np.ones(len(disease_sentence)) # put all the counts at one in Abby's case.
        label = self.label_dict.get(str(eid))
        patient = Patient(disease_sentence, counts_sentence, label)
        
        
        return patient

    def get_genetic_data(self):

        Vcf_file = self.path + f'Chr{self.CHR}/' + "filtered_snps.vcf"

        label_dict = {}

        vcf_reader = vcfpy.Reader(open(Vcf_file, 'r'))
        snp = None
        for record in vcf_reader:
            if record.ID[0] == self.SNP:
                snp = record

        for call in snp.call_for_sample.values():
            sample_id = call.sample.split('_')[0]
            gt_value = call.data.get('GT')

            if gt_value == '0/0':
                label = 0
            elif gt_value == '0/1':
                label = 1
            elif gt_value == '1/1':
                if self.binary_classes:
                    label = 1
                else :
                    label = 2
            else:
                label = None  # Handle other cases if needed
            
            if label is not None:
                label_dict[sample_id] = label
        self.label_dict = label_dict
        return label_dict

    def get_indices_train_test(self, patient_list=None, prop_train_test=0.8):
        if type(self.indices_train) != np.ndarray:
            indices = np.arange(len(patient_list))
            np.random.shuffle(indices)
            self.indices_train = indices[:int(len(patient_list)*prop_train_test)]
            self.indices_test = indices[int(len(patient_list)*prop_train_test):]
        return self.indices_train, self.indices_test



class Patient:
    def __init__(self, diseases_sentence, counts_sentence, label):
        self.diseases_sentence = diseases_sentence
        self.counts_sentence = counts_sentence
        self.SNP_label = label

        self.nb_max_distinct_diseases_patient = None
        self.nb_max_counts_same_disease = None
        self.nb_distinct_diseases = len(self.diseases_sentence)
        self.nb_counts_distinct_diseases = len(self.counts_sentence)

    @property
    def nb_distinct_diseases_actual(self):
        return len(self.diseases_sentence)
    
    @property
    def nb_counts_distinct_diseases_actual(self):
        return len(self.counts_sentence)
    
    def padd_patient(self, nb_max_distinct_disease, padding_item):
        self.diseases_sentence = np.concatenate([self.diseases_sentence, np.full(nb_max_distinct_disease-self.nb_distinct_diseases_actual, padding_item)])
        self.counts_sentence = np.concatenate([self.counts_sentence, np.zeros(nb_max_distinct_disease-self.nb_counts_distinct_diseases_actual)]).astype(int)
        return nb_max_distinct_disease-self.nb_distinct_diseases# to inform on sparcity

    def get_vector(self, nb_max_diseases_sentence):
        patient_grouped = list(zip(self.diseases_sentence, self.counts_sentence))

        # Sort element according to the first list
        patient_group_sorted = sorted(patient_grouped, key=lambda x: x[0])
        # Retrieve the two sorted list
        diseases_sentence_sorted, counts_sentence_sorted = map(np.array, zip(*patient_group_sorted))
        # create patient vector
        vector_patient = np.zeros(nb_max_diseases_sentence)
        vector_patient[diseases_sentence_sorted-1] = counts_sentence_sorted
        self.vector_patient = vector_patient
        return vector_patient

    def get_tree_data(self, nb_max_diseases_sentence):
        res_diseases = np.zeros(nb_max_diseases_sentence)
        res_diseases[self.diseases_sentence] = 1
        return res_diseases
    
class PatientList:
    def __init__(self, list_patients, padding_item=0, pheno_id_dict=None):
        self.patients_list = list_patients
        self.padding_item = padding_item
        self.nb_distinct_diseases_tot = None
        self.nb_max_counts_same_disease = None
        self.nb_max_distinct_diseases_patient = None
        self.label_vector = None
        self.is_padded = False
        self.pheno_id_dict = pheno_id_dict
        self.sparsity = None
        self.seuil = None
        self.share = 1
        

        

    def __len__(self):
        return len(self.patients_list)
    def __getitem__(self, idx):
        return self.patients_list[idx]
    
    def keep_share(self, share):
        if self.share == share:
            pass
        else:
            n = int(share * len(self))
            self.patients_list = self.patients_list[:n]
            self.share = share
    
    def get_nb_max_distinct_diseases_patient(self):
        self.nb_max_distinct_diseases_patient = np.max(np.array([patient.nb_distinct_diseases for patient in self.patients_list]))
        return self.nb_max_distinct_diseases_patient
  
    def get_max_count_same_disease(self):
        self.nb_max_counts_same_disease = max([max(patient.counts_sentence) for patient in self.patients_list])+1 #+1 because of zero counts (padding)
        return self.nb_max_counts_same_disease
    
    def get_nb_distinct_diseases_tot(self):
        self.nb_distinct_diseases_tot = max([max(patient.diseases_sentence) for patient in self.patients_list])+1 #+1 because of the zero padding

        return self.nb_distinct_diseases_tot
    def get_labels_vector(self):
        self.label_vector = np.array([patient.SNP_label for patient in self.patients_list])
        return self.label_vector
    
    def remove_none(self):
        patient_list = []
        for patient in self.patients_list:
            if patient.SNP_label!=None:
                patient_list.append(patient)
        self.patients_list = patient_list


    def get_transformer_data(self, indices_train, indices_test): # transform the patients list in a tuple (distinct_diseases, counts_sentences) list
        transformer_dataset_train = [(patient.diseases_sentence, patient.counts_sentence, patient.SNP_label) for patient in np.array(self.patients_list)[indices_train]]
        transformer_dataset_test = [(patient.diseases_sentence, patient.counts_sentence, patient.SNP_label) for patient in np.array(self.patients_list)[indices_test]]

        return transformer_dataset_train,transformer_dataset_test
    
    def padd_data(self):
        if self.is_padded == True:
            pass
        else:
            if self.nb_max_distinct_diseases_patient==None:
                self.get_nb_max_distinct_diseases_patient()
            padded_data_count = 0
            for patient in self.patients_list:
                padded_data_count += patient.padd_patient(self.nb_max_distinct_diseases_patient, self.padding_item)
            self.sparsity = padded_data_count / (len(self)*self.nb_max_distinct_diseases_patient)
            self.is_padded = True

    def get_matrix_data(self):
        if self.nb_max_distinct_diseases_patient==None:
            nb_distinct_diseases_patient = self.get_nb_max_distinct_diseases_patient()
        patient_list_matrix = np.zeros((len(self),nb_distinct_diseases_patient))
        for i,patient in enumerate(self.patients_list):
            patient_list_matrix[i,:] = patient.get_vector(nb_distinct_diseases_patient)
        
        return patient_list_matrix, self.get_labels_vector()
    
    def get_tree_data(self):
        if self.nb_distinct_diseases_tot==None:
            nb_distinct_diseases_tot = self.get_nb_distinct_diseases_tot()
        label_vector = self.get_labels_vector()
        return ([patient.get_tree_data(self.nb_distinct_diseases_tot) for patient in self.patients_list], label_vector)
    
    def compute_features(self):
        self.get_nb_distinct_diseases_tot()
        self.get_max_count_same_disease()
        self.get_nb_max_distinct_diseases_patient()

    def set_seuil_data(self, seuil):
        new_patient_list = []
        for patient in self.patients_list:
            if patient.nb_counts_distinct_diseases <= seuil:
                new_patient_list.append(patient)
        self.patients_list = new_patient_list
        self.compute_features()

    def get_major_label(self):
        label_vector = self.get_labels_vector()
        nb_zeros = np.sum(label_vector==0)
        nb_ones = np.sum(label_vector==1)
        if nb_zeros >= nb_ones:
            return 0
        else:
            return 1
        
    def equalize_label(self):
        major_label = self.get_major_label()
        nb_minor_label = np.sum(self.label_vector!=major_label)
        counts_major_label = 0
        new_patient_list = []
        for patient in self.patients_list:
            if patient.SNP_label != major_label:
                new_patient_list.append(patient)
            else:
                if counts_major_label < nb_minor_label:
                    new_patient_list.append(patient)
                    counts_major_label +=1
        self.patients_list = new_patient_list
        self.compute_features()

    def split_labels(self):
        patient_label_zero = []
        patient_label_one = []
        for patient in self.patients_list:
            if patient.SNP_label == 0:
                patient_label_zero.append(patient)
            else:
                patient_label_one.append(patient)
        return patient_label_zero, patient_label_one
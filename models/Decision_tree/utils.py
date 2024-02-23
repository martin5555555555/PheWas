import os
import sys
from collections.abc import Iterable

def est_iterable(obj):
    return isinstance(obj, Iterable)

def clear_last_line(output_file):
    with open(output_file, 'r') as file:
        file.seek(0)
        lignes = file.readlines()
        file.close()
    lignes_new = lignes[:-1]
    

    with open(output_file, 'w') as file:
        file.writelines(lignes_new)
        file.close()



def print_file(filename, message, new_line=True):
    with open(filename, 'a') as file:
        if new_line:
            file.write('\n'+message)
        else:
            file.write(message)
        file.close()
def number_tests(model_dir):
    if os.listdir(model_dir) == []:
        return 1
    else:
        last_test_nb = max([ int(test_names.split('_')[0]) for test_names in os.listdir(model_dir)])

        return last_test_nb + 1

def get_name(dataT, pheno_id, decal = True): ## id du pheno in a list with decay
    dec = 1 if decal else 0
    dic_phenos_reverse = {value:key for key, value in dataT.pheno_id_dict.items()}
    if isinstance(pheno_id, Iterable): 
        return [dataT.name_dict[dic_phenos_reverse[i+dec]] for i in pheno_id]
    else:
        return dataT.name_dict[dic_phenos_reverse[pheno_id+dec]]
def get_indice(dataT, names, decal = True):
    dec = 1 if decal else 0
    dic_name_reverse = {value:key for key, value in dataT.name_dict.items()}
    if isinstance(names, Iterable) and not (type(names)== str): 
        return [dataT.pheno_id_dict[dic_name_reverse[name]]-dec for name in names]
    else:
        return dataT.pheno_id_dict[dic_name_reverse[names]] -dec

# Unable bufffering for standard out
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
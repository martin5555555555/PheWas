import os
import sys

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
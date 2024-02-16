print('lancement programme', flush=True)
import sys
path = '/gpfs/commons/groups/gursoy_lab/mstoll/'
sys.path.append(path)

import pandas as pd
import numpy as np
import torch
from functools import partial
import os
import pickle
import time
import copy


from codes.tests.TestsClass import TrainModel, TrainTransformerModel, TestSet
from codes.models.utils import clear_last_line, print_file, number_tests, Unbuffered, plot_infos, plot_ini_infos
from codes.models.data_form.DataForm import PatientList,Patient

# check if an argument is passed 
if len(sys.argv) < 2:
    print("Use: python script_python.py <argument>", flush=True)
    sys.exit(1)

# get the first argument passed to the script
number_instance_test = int(sys.argv[1])

# use the number_instance_test in the script
print("The number of the instance about to be trained is:", number_instance_test, flush=True)



### load the corresponding test
instance_test = TrainModel.load_instance_test(number_instance_test)
instance_test.train_model()
import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np



def Load_mat_single(data_path):

    mat_contents = sio.loadmat(data_path)

    for i, key in enumerate(mat_contents):
        print(i, key)

    return mat_contents

def mat_to_array(mat_contents):
        mat_name = []
        mat_data = []

        for i, (k, v) in enumerate(mat_contents.items()):
            mat_name.append(k) 
            mat_data.append(v)

        vibration_signal_all = np.array(mat_data[3])
        return vibration_signal_all

# Load data
path_good = 'data//baseline_20220915_sv.mat'
path_bad= 'data//fault7_20220915_sv.mat'

mat_contents_good = Load_mat_single(path_good)
good_data = mat_to_array(mat_contents_good)

mat_contents_bad = Load_mat_single(path_bad)
bad_data = mat_to_array(mat_contents_bad)


# constuct the data


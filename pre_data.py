import numpy as np
import os
import h5py

def save_to_hdf5(data, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('dataset', data=data)

def load_data(data_dir):
    data = []
    labels = []
    class_names = []

    for i, file in enumerate(os.listdir(data_dir), start=1):
        if file.endswith('.npy'):
            class_name = file.replace('_data.npy', '')
            class_names.append(class_name)
            npy_path = os.path.join(data_dir, file)
            samples = np.load(npy_path)

            data.append(samples)
            labels.extend([i] * samples.shape[0])

    
    save_to_hdf5(data, "../dataset.h5")

data_dir = "./"

load_data(data_dir)
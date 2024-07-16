import h5py

file_path = './data/MNISTdata.hdf5'

with h5py.File(file_path, 'r') as hdf:
    
    data = hdf

print(data)
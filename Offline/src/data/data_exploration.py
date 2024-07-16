from rich import print

import h5py

import matplotlib.pyplot as plt

DATASET_PATH = "./data/MNISTdata.hdf5"

MNIST_data = h5py.File(DATASET_PATH, 'r')

print(MNIST_data)

print(MNIST_data.keys())

# train data
X_train = MNIST_data['x_train'][:]
Y_train = MNIST_data['y_train'][:]

# val data
X_val = X_train[50000:60000]
Y_val = Y_train[50000:60000]

X_train = X_train[0:50000]
Y_train = Y_train[0:50000]

# test data
X_test = MNIST_data['x_test'][:]
Y_test = MNIST_data['y_test'][:]

print(f"X_train.shape: {X_train.shape}")
print(f"Y_train.shape: {Y_train.shape}")
print(f"X_val.shape: {X_val.shape}")
print(f"Y_val.shape: {Y_val.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"Y_test.shape: {Y_test.shape}")


# for idx in [69, 420, 1337, -1]:
#     image = X_val[idx]
#     label = Y_val[idx]

#     # Reshape the image tensor to a 28x28 matrix
#     image = image.reshape((28, 28))

#     # Plot the image using matplotlib
#     # The image is in grayscale, so we use the 'gray' colormap
#     plt.imshow(image, cmap='gray')
#     # Add a title with the label of the image
#     plt.title(f'MNIST Image (Label: {label.item()})')
#     # Hide the axes for better visualization
#     plt.axis('off')
#     # Display the plot
#     plt.show()

for x, y in zip(X_train, Y_train):
    pass

for x, y in zip(X_val, Y_val):
    pass
for x, y in zip(X_test, Y_test):
    pass

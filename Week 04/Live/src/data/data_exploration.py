from rich import print

import h5py

DATASET_PATH = "./data/MNISTdata.hdf5"

MNIST_data = h5py.File(DATASET_PATH, 'r')

print(MNIST_data)

print(MNIST_data.keys())

X_train = MNIST_data['x_train'][:]
Y_train = MNIST_data['y_train'][:]

X_test = MNIST_data['x_test'][:]
Y_test = MNIST_data['y_test'][:]

print(f"X_train.shape: {X_train.shape}")
print(f"Y_train.shape: {Y_train.shape}")

print(f"X_test.shape: {X_test.shape}")
print(f"Y_test.shape: {Y_test.shape}")

import matplotlib.pyplot as plt

for idx in [10, 20, -1]:
    image = X_train[idx]
    label = Y_train[idx]

    # Reshape the image tensor to a 28x28 matrix
    image = image.reshape((28, 28))

    # Plot the image using matplotlib
    # The image is in grayscale, so we use the 'gray' colormap
    plt.imshow(image, cmap='gray')
    # Add a title with the label of the image
    plt.title(f'MNIST Image (Label: {label})')
    # Hide the axes for better visualization
    plt.axis('off')
    # Display the plot
    plt.show()

for x, y in zip(X_train, Y_train):
    pass
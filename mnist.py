import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import os

def download_mnist_keras(save_path='./mnist_data'):
    """
    Download MNIST dataset using Keras
    """
    print("Downloading MNIST dataset using Keras...")
    
    # Load the dataset 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save as numpy files 
    np.savez(f'{save_path}/mnist_dataset.npz',
             x_train=x_train, y_train=y_train,
             x_test=x_test, y_test=y_test)
    
    print(f"Dataset saved to {save_path}/mnist_dataset.npz")
    print(f"Training set shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = download_mnist_keras()

    x_train=x_train / 255.0
    x_test=x_test / 255.0
    # build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    #compile the model
    model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
    #train model
    print("\nStarting training...")
    model.fit(x_train, y_train, epochs=5)
    #evaluate the model
    print("\nEvaluating on test data:")
    model.evaluate(x_test, y_test, verbose=2)

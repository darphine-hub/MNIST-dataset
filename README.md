# MNIST Handwritten Digit Classification

## Description

This project downloads the MNIST dataset and trains a simple neural network to classify handwritten digits (0-9) using Tensorflow and Keras

## What The Code Does

1. Downloads MNIST datase using Keras
2. Saves dataset locally in `./mnist_data/` folder
3. Normalizes pixel values(0-255 -> 0-1)
4. Builds a neural network with:
   - Flatten layer (input: 28x28)
   - Dense layer(128 neurons, ReLU)
   - Dropout layer(0.2)
   - Dense output layer (10 neurons, Softmax)
5. Compiles model with Adam optimizer and sparse categorical crossentropy loss
6. Trains for 5 epochs
7. Evaluates on test data

## Requirements

```bash
pip install tensorflow numpy keras
```

## Results

```
Epoch 4/5 - accuracy: 0.9739 - loss: 0.0870
Epoch 5/5 - accuracy: 0.9757 - loss: 0.0757

Test results:
- Accuracy: 97.73%
- Loss: 0.0754
```

## Author

- **Name**: Darphine Nyakang'i Mainye
- **Student ID**: CIT-227-066/2023
- **Course**: Foundations of AI

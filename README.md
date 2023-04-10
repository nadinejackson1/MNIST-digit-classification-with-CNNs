# MNIST Digit Classification with Convolutional Neural Networks

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify handwritten digits from the famous MNIST dataset. Previous Exercise: [TensorFlow callbacks using MNIST](https://github.com/nadinejackson1/tensorflow-callbacks-using-mnist) 

The goal is to achieve at least 99.5% accuracy on the training set within 10 epochs.

### Table of Contents

- Introduction
- Data Preprocessing
- Model Architecture
- Training the Model
- Usage

### Introduction

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28x28 pixels in size and in grayscale format. The task is to build a deep learning model that can classify these images with high accuracy.

In this project, we use a Convolutional Neural Network (CNN) to achieve the classification task. CNNs are particularly well-suited for image recognition tasks due to their ability to learn spatial hierarchies of features.
Data Preprocessing

Before feeding the images into the CNN, we need to preprocess the data:

1. Reshape the images to add an extra dimension, representing the color channel (in this case, grayscale). This is done to comply with the standard input format for CNNs, which usually expect 3D tensors for image data.

        images = images.reshape(-1, 28, 28, 1)

2. Normalize the pixel values to be between 0 and 1 by dividing all values by the maximum (255).

        images = images / 255.0

### Model Architecture

The CNN architecture consists of the following layers:

- A Conv2D layer with 32 filters, each with a kernel size of 3x3, and ReLU activation function. The input shape is set to match the shape of each image in the training set (28x28x1).
- A MaxPooling2D layer with a pool size of 2x2, which reduces the spatial dimensions of the feature maps.
- A Flatten layer that converts the 2D feature maps into 1D vectors.
- A Dense layer with 128 units and ReLU activation function.
- A Dense output layer with 10 units and a softmax activation function, representing the 10 possible digit classes.

The model is compiled using the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the evaluation metric.

### Training the Model

To ensure that the training stops when the desired accuracy is achieved (99.5% in this case), a custom Keras callback is implemented:

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.995:
                print("\nReached 99.5% accuracy so cancelling training!")
                self.model.stop_training = True

The model is trained for a maximum of 10 epochs, with the custom callback monitoring the training accuracy and stopping the training process once the target accuracy is reached.

### Usage

To use this project, follow these steps:

    Clone the repository.
    Install the required dependencies (TensorFlow and NumPy).
    Run the Jupyter Notebook, making sure to execute the code blocks in the correct order.

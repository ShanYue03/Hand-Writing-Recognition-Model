# Hand-Writing-Recognition-Model
--------------------------------------------------------------------------------------------------------
This repository contains a Jupyter Notebook (Handwriting_Recognition_NNmodel.ipynb) that demonstrates the implementation of a Neural Network (NN) model for handwriting recognition using the MNIST dataset. The notebook covers the entire workflow, from loading and preprocessing the data to building, training, and evaluating the model. It also includes visualizations of the training results and predictions.

## 📸 Overview
The notebook provides a step-by-step guide to building and training a neural network model for recognizing handwritten digits. The workflow includes:

1. Data Loading and Preprocessing: The MNIST dataset is loaded and normalized to prepare it for training.

2. Model Building: A simple neural network model is constructed using TensorFlow and Keras. The model consists of an input layer, a hidden layer with 128 neurons and ReLU activation, and an output layer with 10 neurons (one for each digit) and softmax activation.

3. Model Compilation: The model is compiled using the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the evaluation metric.

4. Model Training: The model is trained for 5 epochs with 20% of the training data reserved for validation.

5. Model Evaluation: The model's performance is evaluated on the test data, and the test accuracy is reported.

6. Making Predictions: The model is used to make predictions on the test data, and the predicted probabilities for the first test image are displayed.

7. Visualizing Results: The training and validation accuracy and loss are plotted to visualize the model's performance over epochs.

## Key Features
1. Data Preprocessing: The MNIST dataset is normalized to have pixel values between 0 and 1, which helps in faster convergence during training.

2. Model Architecture: The neural network model includes a hidden layer with dropout to prevent overfitting.

3. Training and Validation: The model is trained with a validation split to monitor its performance on unseen data.

4. Visualizations: The notebook includes plots of training and validation accuracy and loss, providing insights into the model's learning process.

5. Prediction Visualization: The notebook demonstrates how to make predictions and visualize the results, including plotting a test image with its predicted label.

## Requirements
To run the notebook, you need the following Python libraries installed:

- TensorFlow

- NumPy

- Matplotlib

# IMDb Sentiment Analysis with LSTM

This project utilizes TensorFlow and Keras to perform sentiment analysis on the IMDb dataset using a Long Short-Term Memory (LSTM) neural network.

## Overview

The goal of this project is to build a binary classification model that predicts the sentiment (positive or negative) of movie reviews from the IMDb dataset.

## Dataset

The IMDb dataset is a popular benchmark for sentiment analysis, containing movie reviews labeled as positive or negative.

## Prerequisites

- Python 3
- TensorFlow
- NumPy
- IMDb dataset (automatically downloaded by Keras)

## Setup

1. Clone the repository:

git clone https://github.com/sona3ms/Week-21.git
cd Week-21


2. Install the required dependencies:

pip install -r requirements.txt

## Usage

1. Run the Jupyter Notebook or Python script to train and evaluate the sentiment analysis model:

python Week-21.py


2. View the training history, test loss, and test accuracy.

## Model Architecture

The neural network architecture consists of the following layers:

- Embedding Layer: Converts integer-encoded words into dense vectors.
- LSTM Layer: Processes sequential information.
- Dense Layer: Produces the final output with a sigmoid activation function for binary classification.

## Hyperparameters

- `max_features`: 10,000
- `max_len`: 500
- `batch_size`: 32
- `epochs`: 5

## Results

The model is trained for 5 epochs, and the test accuracy and loss are printed at the end of the training.

## Acknowledgments

- IMDb for providing the dataset
- TensorFlow and Keras communities for excellent deep learning tools

# Sentiment Analysis detection programme:

## Overview

The marketing initiative to gauge emotional responses to advertisements through facial expressions presents 
an intriguing challenge. In some supermarkets in the UK it is already present and companies test their 
aproducts with a camera to capture the reaction of the customers. This document outlines the conceptual 
framework for developing an emotion detection tool to identify at least three distinct emotional states from 
facial images. This could be done by focusing on categorical emotions such as happiness, surprise, and 
disgust.
The model uses Convolutional Neural Networks (CNNs), involving TensorFlow and Keras due to their 
comprehensive support for deep learning tasks and large libraries. These technologies are chosen for their 
flexibility, extensive documentation, and community support, making them ideal for implementing the 
components of the emotion detection application.


## Key Components

1. **Data Preparation**: 
    - The dataset includes labeled images from the MMAFEDB database, featuring images of real people with various facial expressions. 
    - Images are rescaled and augmented to ensure a diverse dataset in each epoch, using `ImageDataGenerator` from Keras.

2. **Model Architecture**: 
    - A sequential model comprising convolutional layers for feature extraction from facial images, followed by dense layers for classification into emotional states.
    - Regularization techniques like l2 regularization and Dropout are used to reduce overfitting.

3. **Training and Validation**: 
    - The model is trained on a labeled dataset and validated on a separate set to ensure accuracy. 
    - EarlyStopping and ReduceLROnPlateau callbacks are used to optimize the training process.

4. **Evaluation**: 
    - The model's performance is evaluated using accuracy, precision, recall, and F1 score metrics. 
    - Visualization of training results is done using Matplotlib.

## Features

- **Emotion Detection**: Identify emotions such as happiness, surprise, and disgust from facial images.
- **Data Augmentation**: Enhance the dataset with various transformations to improve the model's robustness.
- **Model Regularization**: Techniques like Batch Normalization, Dropout, and l2 regularization to prevent overfitting.
- **Performance Metrics**: Detailed evaluation using accuracy, precision, recall, and F1 score.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <https://github.com/imanjouhar/sentiment_analysis>
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate` instead
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the MMAFEDB dataset**:
    - Place the dataset in the appropriate directory structure and save it as follows, by leaving only
    the sub.folders needed in each of the folders train, valid and test:
    - HAPPY, DISGUST, NEUTRAL, SURPRISE
      ```
      MMAFEDB/
      ├── train/
      ├── valid/
      └── test/
      ```

5. **Run the training script**:
    ```bash
    python sentiment_analysis_june.py
    ```

## Usage

1. **Training the Model**:
    - Ensure the dataset is correctly placed and run the training script. The model will be trained and validated on the provided dataset.

2. **Evaluating the Model**:
    - After training, the performance will be evaluated on the test set. Metrics like accuracy, precision, recall, and F1 score will be printed and visualized.

3. **Predicting Emotions**:
    - Use the trained model to predict emotions on new images. Load the model and pass images to get the predicted emotional states.

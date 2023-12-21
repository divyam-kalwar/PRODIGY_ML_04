# Hand Gesture Recognition using Convolutional Neural Network

This repository contains a Python script for building and training a Convolutional Neural Network (CNN) for hand gesture recognition using the Keras library. The CNN is trained on the LeapGestRecog dataset.

## Dataset
The dataset is located [here](https://www.kaggle.com/datasets/gti-upm/leapgestrecog/code?datasetId=39466&sortBy=voteCount). It contains hand gesture images divided into various categories such as palm, fist, thumb, etc.

## Requirements
- Python 3.x
- Keras
- OpenCV
- Matplotlib
- NumPy
- scikit-learn

## Usage
1. Clone the repository: `git clone https://github.com/divyam-kalwar/PRODIGY_ML_04.git`
2. Navigate to the project directory: `cd PRODIGY_ML_04`
3. Run the Jupyter file: `Hand Gesture.ipynb`

## Model Architecture
The CNN model architecture consists of three convolutional layers with activation functions (ReLU), max-pooling layers, dropout layers, and fully connected layers. The output layer uses the softmax activation function for multiclass classification.

## Training
The model is trained for 11 epochs with a batch size of 64. Training and validation loss/accuracy are plotted, and the best weights are saved during training.

## Evaluation
The trained model is evaluated on the test set, and the accuracy is displayed.

## Files
- `Hand Gesture.ipynb`: Jupyter notebook containing the code for data preprocessing, model creation, training, and evaluation.
- Save model weights after training.

Feel free to customize the code and experiment with different hyperparameters to improve performance.


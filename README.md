# neural-network-challenge-2

# Attrition Prediction Model

This project aims to build a predictive model for employee attrition using data from a CSV file containing various employee attributes. The model utilizes a neural network architecture to predict both attrition and department outcomes.

## Project Overview

The project involves the following steps:

1. Data Preprocessing: The CSV file is loaded and preprocessed to extract relevant features for model training.

2. Model Building: A neural network model is constructed with separate branches for predicting attrition and department outcomes.

3. Model Training: The model is trained on the preprocessed data using categorical cross-entropy loss and softmax activation functions.

4. Model Evaluation: The trained model is evaluated on test data to assess accuracy and performance metrics for both attrition and department predictions.

## Key Questions Addressed

1. Is accuracy the best metric to use on this data? Why or why not?
   - Accuracy may not be the best metric due to class imbalance in the attrition prediction task. Alternative metrics like precision, recall, and F1-score are recommended.

2. What activation functions did you choose for your output layers, and why?
   - Softmax activation functions were used for both output layers to output probabilities for class predictions in a multi-class classification setting.

3. Can you name a few ways that this model might be improved?
   - Feature engineering, model architecture adjustments, optimization, data handling strategies, and advanced techniques can all contribute to improving the model's performance.

## Code Structure

- `preprocessing.ipynb`: Jupyter notebook for data preprocessing steps.
- `model_building.ipynb`: Jupyter notebook for constructing and training the neural network model.
- `evaluation.ipynb`: Jupyter notebook for evaluating the model on test data.

## Dependencies

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the Jupyter notebooks in the specified order to preprocess the data, build the model, and evaluate its performance.



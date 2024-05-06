from double_ml import *
from preprocess import *

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
List defining configurations for experiments.

Variables
---------
model_treatment : sklearn.pipeline.Pipeline | tf.keras.models
    Machine learning model for predicting treatment.
is_model_treatment_dl : bool
    If True, treatment model is a TensorFlow deep learning model.
    If False, scikit-learn model.
model_outcome : sklearn.pipeline.Pipeline | tf.keras.models
    Machine learning model for predicting outcome.
is_model_outcome_dl : bool
    If True, outcome model is a TensorFlow deep learning model.
    If False, scikit-learn model.
model_effect: sklearn.pipeline.Pipeline | tf.keras.models, default=LinearRegression()
    Machine learning model for predicting effect.
is_model_effect_dl : bool, default=False
    If True, effect model is a TensorFlow deep learning model.
    If False, scikit-learn model.
lime_explainer : bool, default=False
    If True, use Lime Explainer.
"""
EXPERIMENTS = [
    {
        'name': 'non-DL treatment (Lasso) + non-DL outcome (Lasso)',
        'task': 'regression',
        'config': {
            'model_treatment': Lasso(),
            'is_model_treatment_dl': False,
            'model_outcome': Lasso(),
            'is_model_outcome_dl': False,
            'model_effect': LinearRegression(),
            'is_model_effect_dl': False,
            'lime_explainer': False
        }
    }, {
        'name': 'non-DL treatment (Lasso) + DL outcome (NN)',
        'task': 'regression',
        'config': {
            'model_treatment': Lasso(),
            'is_model_treatment_dl': False,
            'model_outcome': Sequential([
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_outcome_dl': True
        }
    }, {
        'name': 'DL treatment (NN) + non-DL outcome (Lasso)',
        'task': 'regression',
        'config': {
            'model_treatment': Sequential([
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_treatment_dl': True,
            'model_outcome': Lasso(),
            'is_model_outcome_dl': False
        }
    }, {
        'name': 'DL treatment (NN) + DL outcome (NN)',
        'task': 'regression',
        'config': {
            'model_treatment': Sequential([
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_treatment_dl': True,
            'model_outcome': Sequential([
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_outcome_dl': True
        }   
    }, {
        'name': 'DL treatment (NN) + DL outcome (NN) + DL effect (NN) + lime explainer',
        'task': 'regression',
        'config': {
            'model_treatment': Sequential([
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_treatment_dl': True,
            'model_outcome': Sequential([
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_outcome_dl': True,
            'model_effect': Sequential([
                    Dense(64, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(1)  # Output layer for regression
                ]),
            'is_model_effect_dl': True,
            'lime_explainer': True
        }
    }, {
        'name': 'non-DL treatment (LR) + non-DL outcome (LR)',
        'task': 'regression',
        'config': {
            'model_treatment': LinearRegression(),
            'is_model_treatment_dl': False,
            'model_outcome': LinearRegression(),
            'is_model_outcome_dl': False
        }
        
    }, {
        'name': 'non-DL treatment (RF) + non-DL outcome (RF)',
        'task': 'regression',
        'config': {
            'model_treatment': RandomForestRegressor(),
            'is_model_treatment_dl': False,
            'model_outcome': RandomForestRegressor(),
            'is_model_outcome_dl': False
        }
    }
]

def test_dataset(
    X_train: list[float],
    X_test: list[float],
    T_train: list[float],
    T_test: list[float],
    y_train: list[float],
    y_test: list[float]
) -> None:
    """
    Perform various Double ML experiements on given dataset.
    To edit experiments, change the EXPERIMENTS array.
    Each entry is a dictionary that defines the configurations for the respective experiment.

    Note: Adapted from Alejandro Contreras.

    Parameters
    ----------
    X_train : list[float]
        List containing control training data.
    X_test : list[float]
        List containing control testing data.
    T_train : list[float]
        List containing treatment training data.
    T_test : list[float]
        List containing treatment testing data.
    y_train : list[float]
        List containing target training data.
    y_test : list[float]
        List containing target testing data.

    Returns
    -------
    None
    """

    results = []

    # Conduct each experiment and append results to results array
    for experiment in EXPERIMENTS:
        effect_ceof, mse = double_ml_regression(X_train, X_test, T_train, T_test, y_train, y_test, **experiment['config'])
        results.append([experiment['name'], effect_ceof, mse])
    
    # Convert results array to dataframe and print
    results = np.array(results)
    results_df = pd.DataFrame(results, columns=['experiment','effect coeff','mse'])
    print(results_df)
    return

def main():
    # Run DoubleML experiments on California Housing data
    print('Analyzing California Housing data...')
    X_train, X_test, T_train, T_test, y_train, y_test = preprocess_california_housing()
    test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)

    # Run DoubleML experiments on Communities and Crime data
    print('Analyzing Communities and Crime data...')
    X_train, X_test, T_train, T_test, y_train, y_test = preprocess_communities_and_crime()
    test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)
    
    # # Run experiments on Education data
    # print('Analyzing Education data...')
    # X_train, X_test, T_train, T_test, y_train, y_test = preprocess('data/statedata.csv', 'Illiteracy', 'HS.Grad')
    # test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)

    # # Run experiments on Education data
    # print('Analyzing Education data...')
    # X_train, X_test, T_train, T_test, y_train, y_test = preprocess('data/statedata.csv', 'Income', 'HS.Grad')
    # test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)

if __name__ == '__main__':
    main()
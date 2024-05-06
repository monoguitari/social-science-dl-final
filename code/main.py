from double_ml import double_ml
from preprocess import preprocess, preprocess_california_housing

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense

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
    To edit experiments, change the experiments array.
    Each entry is a dictionary that defines the configurations for the respective experiment.
    See double_ml in double_ml.py for variable definitions.

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

    # Initiate DL neural network for treatment and outcome models
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Initate DL neural network for effect model
    nn_model_effect = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
    ])

    # Define configurations for experiments
    experiments = [
        {
            'name': 'non-DL treatment (Lasso) + non-DL outcome (Lasso)',
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
            'config': {
                'model_treatment': Lasso(),
                'is_model_treatment_dl': False,
                'model_outcome': clone_model(nn_model),
                'is_model_outcome_dl': True
            }
        }, {
            'name': 'DL treatment (NN) + non-DL outcome (Lasso)',
            'config': {
                'model_treatment': clone_model(nn_model),
                'is_model_treatment_dl': True,
                'model_outcome': Lasso(),
                'is_model_outcome_dl': False
            }
        }, {
            'name': 'DL treatment (NN) + DL outcome (NN)',
            'config': {
                'model_treatment': clone_model(nn_model),
                'is_model_treatment_dl': True,
                'model_outcome': clone_model(nn_model),
                'is_model_outcome_dl': True
            }   
        }, {
            'name': 'DL treatment (NN) + DL outcome (NN) + DL effect (NN) + lime explainer',
            'config': {
                'model_treatment': clone_model(nn_model),
                'is_model_treatment_dl': True,
                'model_outcome': clone_model(nn_model),
                'is_model_outcome_dl': True,
                'model_effect': clone_model(nn_model_effect),
                'is_model_effect_dl': True,
                'lime_explainer': True
            }
        }, {
            'name': 'non-DL treatment (LR) + non-DL outcome (LR)',
            'config': {
                'model_treatment': LinearRegression(),
                'is_model_treatment_dl': False,
                'model_outcome': LinearRegression(),
                'is_model_outcome_dl': False
            }
            
        }, {
            'name': 'non-DL treatment (RF) + non-DL outcome (RF)',
            'config': {
                'model_treatment': RandomForestRegressor(),
                'is_model_treatment_dl': False,
                'model_outcome': RandomForestRegressor(),
                'is_model_outcome_dl': False
            }
        }
    ]

    # Conduct each experiment and append results to results array
    for experiment in experiments:
        effect_ceof, mse = double_ml(X_train, X_test, T_train, T_test, y_train, y_test, **experiment['config'])
        results.append([experiment['name'], effect_ceof, mse])
    
    # Convert results array to dataframe and print
    results = np.array(results)
    results_df = pd.DataFrame(results, columns=['experiment','effect coeff','mse'])
    print(results_df)
    return

    """
    OLD CODE

    # Perform doubleML analysis w/ non-DL treatment model (Lasso), non-DL outcome model (Lasso), non-DL effect model (linear regression)
    print('__DoubleML w/ non-DL treatment model (Lasso), non-DL outcome model (Lasso)__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, Lasso(), False, Lasso(), False)

    # Perform doubleML analysis w/ non-DL treatment model (Lasso), DL outcome model (neural network), non-DL effect model (linear regression)
    print('__DoubleML w/ non-DL treatment model (Lasso), DL outcome model (NN)__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, Lasso(), False, clone_model(nn_model), True)

    # Perform doubleML analysis w/ DL treatment model (neural network), non-DL outcome model (Lasso), non-DL effect model (linear regression)
    print('__DoubleML w/ DL treatment model (NN), non-DL outcome model (Lasso)__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, clone_model(nn_model), True, Lasso(), False)

    # Perform doubleML analysis w/ DL treatment model (neural network), DL outcome model (neural network), non-DL effect model (linear regression)
    print('__DoubleML w/ DL treatment model (NN), DL outcome model (NN)__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, clone_model(nn_model), True, clone_model(nn_model), True)   

    # TODO: Seems like there is bug in notebook code, did not initate y_test_pred, T_resid_test_array in corresp. cell.
    # Perform doubleML analysis w/ DL treatment model (neural network), DL outcome model (neural network), DL effect model (neural network), Lime Explainer
    print('__DoubleML w/ DL treatment model (NN), DL outcome model (NN), DL effect model (NN), lime explainer__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, clone_model(nn_model), True, clone_model(nn_model), True,
            model_effect=clone_model(nn_model_effect), is_model_effect_dl=True, lime_explainer=True)   
    
    # Perform doubleML analysis w/ non-DL treatment model (linear regression), non-DL outcome model (linear regression), non-DL effect model (linear regression)
    print('__DoubleML w/ non-DL treatment model (LR), non-DL outcome model (LR)__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, LinearRegression(), False, LinearRegression(), False)   

    # Perform doubleML analysis w/ non-DL treatment model (RF), non-DL outcome model (RF), non-DL effect model (linear regression)
    print('__Performing doubleML analysis w/ non-DL treatment model (RF), non-DL outcome model (RF)__')
    double_ml(X_train, X_test, T_train, T_test, y_train, y_test, RandomForestRegressor(), False, RandomForestRegressor(), False)
    """

def main():
    # Run DoubleML experiments on California Housing data
    print('Analyzing California Housing data...')
    X_train, X_test, T_train, T_test, y_train, y_test = preprocess_california_housing()
    test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)

    # Run experiments on Education data
    print('Analyzing Education data...')
    X_train, X_test, T_train, T_test, y_train, y_test = preprocess('data/statedata.csv', 'Illiteracy', 'HS.Grad')
    test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)

    # Run experiments on Education data
    print('Analyzing Education data...')
    X_train, X_test, T_train, T_test, y_train, y_test = preprocess('data/statedata.csv', 'Income', 'HS.Grad')
    test_dataset(X_train, X_test, T_train, T_test, y_train, y_test)


if __name__ == '__main__':
    main()
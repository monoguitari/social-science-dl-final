from __future__ import annotations

import numpy as np
import random
import os
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from keras import backend as K

from lime.lime_tabular import LimeTabularExplainer

def reset_random_seeds(seed_value=1):
    """
    Reset random seeds and TensorFlow session for reproducibility.
    
    Parameters
    ----------
    seed_value: int, default 42. The seed value to use for all random operations.

    Note: Adapted from https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    """
    # Set PYTHONHASHSEED environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # Set python built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    
    # Set numpy pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    
    # Set tensorflow pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    
    # Configure a new global tensorflow session
    # Check if TensorFlow 2.x is used, which is determined by the presence of tf.compat module
    if hasattr(tf, 'compat'):
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                                inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
    else:  # TensorFlow 1.x
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

def double_ml(
    X_train: list[float],
    X_test: list[float],
    T_train: list[float],
    T_test: list[float],
    y_train: list[float],
    y_test: list[float],
    model_treatment: sklearn.pipeline.Pipeline | tf.keras.models,
    is_model_treatment_dl: bool,
    model_outcome: sklearn.pipeline.Pipeline | tf.keras.models,
    is_model_outcome_dl: bool,
    model_effect: sklearn.pipeline.Pipeline | tf.keras.models = LinearRegression(),
    is_model_effect_dl: bool = False,
    lime_explainer: bool = False
) -> tuple[float, float]:
    """
    Performs double machine learning analysis.
    
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
    model_treatment : sklearn.pipeline.Pipeline | tf.keras.models
        Machine learning model for predicting treatment.
    is_model_treatment_dl : bool
        If True, treatment model is a TensorFlow deep learning model.
    model_outcome : sklearn.pipeline.Pipeline | tf.keras.models
        Machine learning model for predicting outcome.
    is_model_outcome_dl : bool
        If True, outcome model is a TensorFlow deep learning model.
    model_effect: sklearn.pipeline.Pipeline | tf.keras.models, default=LinearRegression()
        Machine learning model for predicting effect.
    is_model_effect_dl : bool, default=False
        If True, effect model is a TensorFlow deep learning model.
    lime_explainer : bool, default=False
        If True, use Lime Explainer.

    Returns
    -------
    effect_estimate : float
        Estimated effect of treatment variable on target variable.
    mse : float
        Test mean squared error.
    """
    # Reset random seed
    reset_random_seeds(seed_value=1)

    # Fitting treatment model
    if is_model_treatment_dl:
        model_treatment.compile(optimizer='adam', loss='mse')
        model_treatment.fit(X_train, T_train, epochs=50, batch_size=32, verbose=0)
    else:
        model_treatment.fit(X_train, T_train)
    
    # Fitting outcome model
    if is_model_outcome_dl:
        model_outcome.compile(optimizer='adam', loss='mse')
        model_outcome.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    else:
        model_outcome.fit(X_train, y_train)

    # Predicting and calculating residuals
    T_train_pred = model_treatment.predict(X_train)
    y_train_pred = model_outcome.predict(X_train)
    if is_model_treatment_dl:
        T_train_pred = T_train_pred.flatten()
    if is_model_outcome_dl:
        y_train_pred = y_train_pred.flatten()

    T_resid = T_train - T_train_pred
    y_resid = y_train - y_train_pred

    # Convert to numpy array for reshaping
    T_resid_array = T_resid.to_numpy().reshape(-1, 1)
    y_resid_array = y_resid.to_numpy()

    # Estimating treatment effect using residuals
    if is_model_effect_dl:
        model_effect.compile(optimizer='adam', loss='mse')
        model_effect.fit(T_resid_array, y_resid_array, epochs=50, batch_size=32, verbose=0)
    else:
        model_effect.fit(T_resid_array, y_resid)

    # Evaluate the model on the test set
    T_test_pred = model_treatment.predict(X_test)
    y_test_pred = model_outcome.predict(X_test)
    if is_model_treatment_dl:
        T_test_pred = T_test_pred.flatten()
    if is_model_outcome_dl:
        y_test_pred = y_test_pred.flatten()

    T_resid_test = T_test - T_test_pred
    y_resid_test = y_test - y_test_pred

    # Convert to numpy array for reshaping
    T_resid_test_array = T_resid_test.to_numpy().reshape(-1, 1)

    # Predict the effect
    effect_estimate = model_effect.predict(T_resid_test_array)
    if is_model_effect_dl:
        effect_estimate_coeff = model_effect.layers[-1].get_weights()[0][0][0]
        # effect_estimate = effect_estimate.flatten()
    else:
        effect_estimate_coeff = model_effect.coef_[0]
    # print("Estimated effect of treatment variable on target variable:", effect_estimate_coeff)

    # Evaluating model performance
    if is_model_effect_dl:
        test_predictions = y_test_pred + effect_estimate.flatten() * T_resid_test
    else:
        test_predictions = y_test_pred + effect_estimate * T_resid_test
    mse = mean_squared_error(y_test, test_predictions)
    # print("Test MSE:", mse)

    if lime_explainer:
        # Convert DataFrame to numpy array if not already done
        X_train_np = X_train.to_numpy().astype(np.float32)
        X_test_np = X_test.to_numpy().astype(np.float32)

        # Creating a LIME explainer for the model
        explainer = LimeTabularExplainer(X_train_np, mode='regression', feature_names=X_train.columns.tolist(), categorical_features=[], class_names=['Value'])
        explanation = explainer.explain_instance(X_test_np[0], model_outcome.predict, num_features=5)
        explanation.show_in_notebook(show_table=True)

    return effect_estimate_coeff, mse
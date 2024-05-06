from __future__ import annotations

import numpy as np
import random
import os
import sklearn

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, log_loss

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

def double_ml_regression(
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
    Performs double machine learning analysis for regression tasks.
    
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

''''
def double_ml_classification(
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
    model_effect: sklearn.pipeline.Pipeline | tf.keras.models = LogisticRegression(),
    is_model_effect_dl: bool = False,
    lime_explainer: bool = False
) -> tuple[float, float]:
    # Reset random seeds for consistent results
    reset_random_seeds(seed_value=1)

    # Convert sparse matrices to dense arrays if necessary
    X_train_dense = X_train.toarray() if 'toarray' in dir(X_train) else X_train
    X_test_dense = X_test.toarray() if 'toarray' in dir(X_test) else X_test

    # Logistic Regression for predicting treatment
    model_treatment = LogisticRegression()
    if is_model_treatment_dl:
        pass
    else:
        model_treatment.fit(X_train_dense, T_train)
        T_train_pred_prob = model_treatment.predict_proba(X_train_dense)[:, 1]

    # Evaluate Logistic Regression Model
    accuracy_treatment = accuracy_score(T_train, T_train_pred_prob > 0.5)
    f1_treatment = f1_score(T_train, T_train_pred_prob > 0.5)
    log_loss_treatment = log_loss(T_train, T_train_pred_prob)

    print("Training Accuracy (Treatment model):", accuracy_treatment)
    print("Training F1 Score (Treatment model):", f1_treatment)
    print("Training Log Loss (Treatment model):", log_loss_treatment)
    print()

    # Neural network for predicting outcome
    model_outcome = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_dense.shape[1],)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for classification
    ])
    model_outcome.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model_outcome.fit(X_train_dense, y_train, epochs=50, batch_size=32, verbose=0)

    y_train_pred_prob = model_outcome.predict(X_train_dense).flatten()

    # Evaluate Neural Network Model
    accuracy_outcome = accuracy_score(y_train, y_train_pred_prob > 0.5)
    f1_outcome = f1_score(y_train, y_train_pred_prob > 0.5)
    log_loss_outcome = log_loss(y_train, y_train_pred_prob)

    print("Training Accuracy (Outcome model):", accuracy_outcome)
    print("Training F1 Score (Outcome model):", f1_outcome)
    print("Training Log Loss (Outcome model):", log_loss_outcome)
    print()

    # Calculating residuals
    T_resid = T_train - T_train_pred_prob
    y_resid = y_train - y_train_pred_prob

    # Convert residuals to numpy arrays before reshaping
    T_resid_array = T_resid.to_numpy().reshape(-1, 1)
    y_resid_array = (y_resid.to_numpy() > 0.5).astype(int)

    # Logistic Regression to estimate the treatment effect using residuals
    model_effect = LogisticRegression()
    model_effect.fit(T_resid_array, y_resid_array)

    # Print the estimated effect size
    print("Estimated effect size of the treatment variable:", model_effect.coef_[0])
    print()

    # Predictions for the test set
    T_test_pred_prob = model_treatment.predict_proba(X_test_dense)[:, 1]
    y_test_pred_prob = model_outcome.predict(X_test_dense).flatten()

    # Calculate test residuals
    T_resid_test = T_test - T_test_pred_prob
    y_resid_test = y_test - y_test_pred_prob

    # Convert test residuals to numpy arrays
    T_resid_test_array = T_resid_test.to_numpy().reshape(-1, 1)

    # Predict the effect on the test set using logistic regression
    effect_estimate_prob = model_effect.predict_proba(T_resid_test_array)[:, 1]

    # Combining predicted effect with the outcome predictions to generate final predictions
    test_predictions_prob = y_test_pred_prob + effect_estimate_prob
    test_predictions = (test_predictions_prob > 0.5).astype(int)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)
    log_loss_value = log_loss(y_test, test_predictions_prob)

    print("Test Accuracy:", accuracy)
    print("Test F1 Score:", f1)
    print("Test Log Loss:", log_loss_value)

    return accuracy, f1, log_loss_value
'''
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def preprocess_california_housing(T_col: str = 'HouseAge', y_col: str = 'MedHouseVal'
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
    Preprocess California Housing data and return data split into training and testing sets.

    Parameters
    ----------
    T_col : str
        String corresponding to column of treatment variable
    y_col : str
        String corresponding to column of target variable

    Returns
    -------
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
    """
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)

    # Convert to DataFrame
    data = housing.frame

    # Show the first few rows of the dataframe
    # print(data.head())

    # Selecting the target and the explanatory variable
    y = data[y_col]
    T = data[T_col]
    X = data.drop([y_col, T_col], axis=1)  # Exclude target and treatment for control features

    # Splitting the data into training and testing sets
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.2, random_state=42)

    return X_train, X_test, T_train, T_test, y_train, y_test

def preprocess(file: str, T_col: str, y_col: str
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
    Preprocess CSV file and return data split into training and testing sets.

    Parameters
    ----------
    file: str
        Filepath of CSV file.
    T_col : str
        String corresponding to column of treatment variable.
    y_col : str
        String corresponding to column of target variable.

    Returns
    -------
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
    """

    # Read CSV filepath and convert to DataFrame
    data = pd.read_csv(file)

    # Select only numerical columns
    # TODO: one-hot encode non-numerical columns (?)
    data = data.select_dtypes(['number'])
    
    #print(data.head())

    # Selecting the target and the explanatory variable
    y = data[y_col]
    T = data[T_col]
    X = data.drop([y_col, T_col], axis=1)  # Exclude target and treatment for control features

    # Splitting the data into training and testing sets
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.2, random_state=42)

    return X_train, X_test, T_train, T_test, y_train, y_test
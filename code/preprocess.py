from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 

def preprocess_california_housing(T_col: str = 'HouseAge', y_col: str = 'MedHouseVal'
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """
    Preprocess California Housing data and return data split into training and testing sets.

    Parameters
    ----------
    T_col : str, default='HouseAge'
        String corresponding to column of treatment variable
    y_col : str, default='MedHouseVal'
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

def preprocess_communities_and_crime():
    # Fetch the dataset 
    communities_and_crime = fetch_ucirepo(id=183) 

    # Data (as pandas dataframes) 
    X = communities_and_crime.data.features 
    y = communities_and_crime.data.targets 

    # Metadata (uncomment to print)
    # print(communities_and_crime.metadata) 

    # Adjust display settings to show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Variable information
    print(communities_and_crime.variables)

    # Fetch the dataset
    communities_and_crime = fetch_ucirepo(id=183)

    # Data (as pandas dataframes)
    X = communities_and_crime.data.features
    y = communities_and_crime.data.targets

    # Keep only continuous features
    continuous_columns = [col for col, dtype in zip(X.columns, communities_and_crime.variables['type']) if dtype == 'Continuous']
    X_continuous = X[continuous_columns]

    # Convert '?' to NaN
    X_continuous = X_continuous.replace('?', np.nan)

    # Convert all columns to numeric, if not already
    X_continuous = X_continuous.apply(pd.to_numeric, errors='coerce')

    # Impute missing values with the mean of each column
    X_continuous.fillna(X_continuous.mean(), inplace=True)

    # Calculate correlation matrix
    corr_matrix = X_continuous.corr().abs()  # Absolute value of correlation coefficients

    # Exclude 'PctPopUnderPov' from high correlation removal
    relevant_columns = set(X_continuous.columns) - {'PctPopUnderPov'}

    # Identify highly correlated features (correlation > 0.8) excluding 'PctPopUnderPov'
    high_corr_var = [col for col in relevant_columns if any((corr_matrix[col] > 0.8) & (corr_matrix.index != col))]

    # Drop highly correlated features
    X = X_continuous.drop(columns=high_corr_var, errors='ignore')

    # Define target variable `y` and treatment variable
    y = y['ViolentCrimesPerPop']
    T = X['PctPopUnderPov']

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.2, random_state=42)

    # Display the resulting data
    # print(X.head())
    # print(y.head())
    # print(T.head())

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
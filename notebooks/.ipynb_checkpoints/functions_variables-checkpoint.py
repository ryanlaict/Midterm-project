import pandas as pd
import numpy as np
import re
import ast
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, KFold

def encode_tags(df):
    """
    One-hot encode the 'tags' column.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'tags' column.

    Returns:
        pandas.DataFrame: Modified with encoded tag columns.
    """
    # Convert lists to comma-separated strings
    df["tags"] = df["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # Apply one-hot encoding
    return df.join(df["tags"].str.get_dummies(sep=", ")).drop(columns=["tags"])

def encode_primary_photo(df):
    """
    Converts the 'primary_photo' column into a binary (True/False) column 
    based on whether an href exists.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'primary_photo'.

    Returns:
        pandas.DataFrame: Modified DataFrame with 'has_primary_photo' column.
    """
    # Create a new binary column: True if 'href' exists, False otherwise
    df['has_primary_photo'] = df['primary_photo'].apply(lambda x: isinstance(x, dict) and 'href' in x)

    # Drop the original 'primary_photo' column
    df = df.drop(columns=['primary_photo'])

    return df

def encode_source(df):
    """
    Creates two new binary columns:
    - 'agent': 1 if 'agents' exist and are not None, 0 otherwise.
    - 'mls': 1 if 'type' exists and is not None, 0 otherwise.
    
    Then, drops the original 'source' column.
    """
    # Convert string-based dictionaries to actual dictionaries
    df['source'] = df['source'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Check for 'agents' and 'type' keys
    df['agent'] = df['source'].apply(lambda x: 1 if isinstance(x, dict) and x.get('agents') else 0)
    df['mls'] = df['source'].apply(lambda x: 1 if isinstance(x, dict) and x.get('type') else 0)

    # Drop the original 'source' column
    df.drop(columns=['source'], inplace=True)

    return df

def extract_city_state(df):
    """
    Extracts City and State from the 'permalink' column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'permalink'.

    Returns:
        pandas.DataFrame: Modified DataFrame with 'city' and 'state' columns.
    """

    # Function to extract City
    def get_city(permalink):
        match = re.search(r'_(.*?)_[A-Z]{2}_', permalink)
        return match.group(1) if match else None

    # Function to extract State
    def get_state(permalink):
        match = re.search(r'_([A-Z]{2})_\d{5}_', permalink)
        return match.group(1) if match else None

    # Apply extraction functions to permalink column
    df['city'] = df['permalink'].apply(lambda x: get_city(x) if isinstance(x, str) else None)
    df['state'] = df['permalink'].apply(lambda x: get_state(x) if isinstance(x, str) else None)

    # Drop the original 'permalink' column
    df.drop(columns=['permalink'], inplace=True)
    
    return df

def categorize_cost_of_living(score):
    if score < 100:
        return 'Low'
    elif 100 <= score <= 120:
        return 'Average'
    elif 120 < score <= 150:
        return 'High'
    else:
        return 'Very High'

def evaluate_model(model, X_test, y_test):
    """Prints MAE, RMSE, and R² for a given model"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{model.__class__.__name__}:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}\n")

def evaluate_poly_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nPolynomial Regression Model - {dataset_name} Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

def custom_cross_validation(X_train, y_train, n_splits=5):
    '''Creates n_splits sets of training and validation folds using K-Fold cross-validation.

    Args:
      training_data (pd.DataFrame): The dataframe of features and target to be divided into folds.
      n_splits (int): The number of sets of folds to be created.

    Returns:
      tuple: A tuple of lists, where the first index is a list of the training folds, 
             and the second index is the corresponding validation folds.

    Example:
        >>> output = custom_cross_validation(train_df, n_splits=10)
        >>> output[0][0] # The first training fold
        >>> output[1][0] # The first validation fold
        >>> output[0][1] # The second training fold
        >>> output[1][1] # The second validation fold... etc.
    '''
    training_data = pd.concat([X_train, y_train], axis=1)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Shuffle
    train_folds = []
    val_folds = []

    for train_index, val_index in kfold.split(training_data):
        train_fold = training_data.iloc[train_index] 
        val_fold = training_data.iloc[val_index]  
        train_folds.append(train_fold)
        val_folds.append(val_fold)

    return train_folds, val_folds

def hyperparameter_search(training_folds, validation_folds, param_grid, model, scoring=mean_squared_error, higher_is_better=False):
    '''
    Performs a custom grid search for the best hyperparameters using k-fold validation.
    
    Args:
      training_folds (list): List of training fold dataframes (features and target concatenated).
      validation_folds (list): List of validation fold dataframes (features and target concatenated).
      param_grid (dict): Dictionary of possible hyperparameter values.
      model: Model that will be used to fit.
      scoring (function): Scoring function to evaluate model performance. Default is mean_squared_error.
      higher is better (bool): If True, higher scores are better; if False, lower scores are better. Default is False. This is to take into account R2 where the larger the score is better
      
    Returns:
      dict: Best hyperparameter settings based on the chosen metric.
    '''
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    best_score = float('-inf') if higher_is_better else float('inf')
    best_params = None
    
    for combination in param_combinations:
        params = dict(zip(param_names, combination))
        scores = []
        # print(f"Testing parameters: {params}") # Comment this in if you want to see parameters used at every set of parameters
        
        for train_fold, val_fold in zip(training_folds, validation_folds):
            X_train, y_train = train_fold.iloc[:, :-1], train_fold.iloc[:, -1]
            X_val, y_val = val_fold.iloc[:, :-1], val_fold.iloc[:, -1]
            
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            score = scoring(y_val, predictions)
            scores.append(score)
        
        avg_score = np.mean(scores)
        # print(f"Average Score: {avg_score:.4f}\n") # Comment this in if you want to see average score for every set of parameters
        
        if (higher_is_better and avg_score > best_score) or (not higher_is_better and avg_score < best_score):
            best_score = avg_score
            best_params = params
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score:.4f}")
    
    return best_params
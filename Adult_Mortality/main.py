import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
import joblib

# File paths
TRAIN_DATA_PATH = './data/train_data.csv'
MODEL_FILENAME = './model.pkl'
IMPUTER_FILENAME = './imputer.pkl'
SCALER_FILENAME = './scaler.pkl'

# Column names
COLUMN_NAMES = [
    'Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
    'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
    'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
    ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
    'Schooling', 'Adult Mortality'
]


def preprocess_data(data, imputer=None, scaler=None):
    # Drop non-numeric columns
    data = data.drop(["Country", "Status"], axis=1)

    # Impute missing values if imputer is not provided
    if imputer is None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = imputer.fit(data[COLUMN_NAMES[:-1]])
    data[COLUMN_NAMES[:-1]] = imputer.transform(data[COLUMN_NAMES[:-1]])

    # Scale data if scaler is not provided
    if scaler is None:
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)

    # Drop the 'Year' column
    data_norm = data_norm.drop(['Year'], axis=1)

    return data_norm, imputer, scaler


def model_fit(model_name, train_x, train_y):
    # Define parameter grids for each model
    param_grids = {
        'MLPRegressor': {
            'hidden_layer_sizes': [(100, 100), (200, 200)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
        },
        'RandomForestRegressor': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'AdaBoostRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1]
        }

    }

    # Initialize the regressor based on the model name
    if model_name in param_grids:
        regressor = eval(model_name)(max_iter=1000)
        param_grid = param_grids[model_name]
        regressor = GridSearchCV(regressor, param_grid, cv=5, scoring='r2', n_jobs=5)
    else:
        regressor = eval(model_name)(max_iter=1000)

    # Fit the model
    regressor.fit(train_x, train_y)
    return regressor


def predict(model, test_data, imputer, scaler):
    # Preprocess the test data
    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values

    # Make predictions
    predictions = model.predict(test_x)
    return predictions


def main():
    # Load training data
    train_data = pd.read_csv(TRAIN_DATA_PATH)

    # Initialize KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_r2 = -np.inf
    best_imputer = None
    best_scaler = None
    best_model_name = None

    # List of models to evaluate
    model_list = ["MLPRegressor", "RandomForestRegressor", "AdaBoostRegressor", "DecisionTreeRegressor", "Ridge", "SVR"]

    # Iterate over each model
    for model_name in model_list:
        for train_index, test_index in kf.split(train_data):
            # Split data into training and testing folds
            train_fold = train_data.iloc[train_index]
            test_fold = train_data.iloc[test_index]

            # Separate target variable
            train_y = train_fold['Adult Mortality'].values
            train_fold = train_fold.drop(["Adult Mortality"], axis=1)

            # Preprocess training data
            train_fold_norm, imputer, scaler = preprocess_data(train_fold)
            train_x = train_fold_norm.values

            # Fit the model
            model = model_fit(model_name, train_x, train_y)

            # Separate target variable for testing data
            test_y = test_fold['Adult Mortality'].values
            test_fold = test_fold.drop(["Adult Mortality"], axis=1)

            # Make predictions
            y_pred = predict(model, test_fold, imputer, scaler)

            # Calculate R2 score for testing data
            r2 = r2_score(test_y, y_pred)

            # Calculate R2 score for training data
            train_pred = model.predict(train_x)
            train_r2 = r2_score(train_y, train_pred)
            print(f"Model name: {model_name}, Fold Train R2: {train_r2}, Test R2: {r2}")

            # Update the best model if current model is better
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = model_name
                best_imputer = imputer
                best_scaler = scaler
                print(f"Best model updated: {model_name}, R2: {best_r2}")

    # Save the best model, imputer, and scaler
    joblib.dump(best_model, MODEL_FILENAME)
    joblib.dump(best_imputer, IMPUTER_FILENAME)
    joblib.dump(best_scaler, SCALER_FILENAME)
    print(f'Best model: {best_model_name}, R2: {best_r2}')


if __name__ == "__main__":
    main()

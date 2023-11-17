'''
A project to compare models for predictive analysis of Formula One racing
'''

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold, cross_val_predict

from sklearn.metrics import mean_absolute_error, r2_score, f1_score

# Import project-specific helper functions
from clean_data import read_raw_data, merge_raw_data, drop_missing_observations
from feature_engineering import get_quali_times, add_driver_age, add_lead_driver, \
                                add_points_difference_to_teammate
from analysis_metrics import regression_metrics, classifier_metrics
from visualizations import feature_importances, predictions_v_true, confusion_matrix

def main():
    # Read in data
    files = read_raw_data()
    f1_data = merge_raw_data(files)

    # Filter data from 2011 onwards
    f1_data = f1_data.loc[f1_data['date'].dt.year >= 2011]

    # Perform feature engineering
    f1_data = get_quali_times(f1_data)
    f1_data = add_driver_age(f1_data)
    f1_data = add_lead_driver(f1_data)
    f1_data = add_points_difference_to_teammate(f1_data)

    # Set the index to date
    f1_data.set_index('date', inplace=True)
    f1_data['date'] = f1_data.index

    # Remove ID information
    f1_data.drop(columns=['raceId', 'circuitId', 'driverId', 'constructorId'], inplace=True)

    # Drop missing observations
    f1_data = drop_missing_observations(f1_data)

    # Rearrange columns for readability
    f1_data = f1_data[['round', 'circuitRef', 'alt', 'constructorRef', 'driverRef', 'lead_driver',
                       'points_to_teammate', 'table_position', 'age', 'nationality', 'quali_time',
                       'grid', 'avg_lap', 'stops', 'position']]

    # Correct datatypes to integers
    int_cols = ['alt', 'table_position', 'quali_time', 'grid', 'avg_lap', 'stops', 'position']
    f1_data[int_cols] = f1_data[int_cols].astype('int32')


    ### model setup

    # Define the target variable and features
    TARGET = 'position'
    y = f1_data[TARGET]
    X = f1_data.drop(TARGET, axis=1)

    # Set up K-Fold Cross Validation with six two-year increment splits (2014 to 2023)
    six_fold_cv = KFold(n_splits=5)


    ### Regression models
    
    # calculate baseline metrics
    base_model = [y.mean()] * len(y)
    base_mae = mean_absolute_error(y, base_model)
    base_r2 = r2_score(y, base_model)

    metrics = pd.DataFrame({'regression model': ['baseline'],
                            'MAE': [base_mae],
                            'r2 score': [base_r2]})

    # define categorical and numerical columns for column transformer
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()

    # create list of Regression models to loop thru
    regression_models = [LinearRegression(), Ridge(), Lasso()]

    # model generator loop
    for model_type in regression_models:
        model = make_pipeline(
            ColumnTransformer(
                transformers=[
                    # One-hot encode the string columns
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                    # Include numerical columns without any transformation
                    ('num', 'passthrough', numerical_columns)
                ]),
            SimpleImputer(),
            StandardScaler(with_mean=False),
            model_type
        )
        model.fit(X, y)

        # add model metrics to metrics table
        metrics.loc[len(metrics)] = regression_metrics(model, X, y, six_fold_cv)

        # generate visualizations
        feature_importances(model)

        y_pred = cross_val_predict(model, X, y, cv=six_fold_cv)
        predictions_v_true(y, y_pred)
    
    print(metrics, '\n')


    ### classification models

    # calculate baseline metrics
    base_model = [y.mean()] * len(y)
    base_mae = mean_absolute_error(y, base_model)
    base_acc = y.value_counts(normalize=True).max()
    base_f1 = f1_score(y, [1]*len(y), average="macro")

    metrics = pd.DataFrame({'classifier model': ['baseline'],
                            'MAE': [base_mae],
                            'accuracy': [base_acc],
                            'f1': [base_f1]})

    # list of classifier models to loop thru
    classifier_models = [DecisionTreeClassifier(min_samples_split=69,
                                                min_samples_leaf=6,
                                                max_depth=63,
                                                criterion='entropy',
                                                random_state=34),
                         RandomForestClassifier(n_estimators=134,
                                                min_samples_split=69,
                                                min_samples_leaf=6,
                                                max_depth=63,
                                                criterion='entropy',
                                                random_state=34),
                         XGBClassifier(random_state=34)]
    
    # model generator loop
    for model_type in classifier_models:
        model = make_pipeline(
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999999),
                SimpleImputer(),
                model_type)
        
        # XG model must be zero-indexed
        if model_type == classifier_models[-1]:
            y = y - 1
        
        model.fit(X, y)
        
        # add model metrics to table
        metrics.loc[len(metrics)] = classifier_metrics(model, X, y, six_fold_cv)

        # generate visualizations
        feature_importances(model)

        y_pred = cross_val_predict(model, X, y, cv=six_fold_cv)
        confusion_matrix(model, y, y_pred)

    
    print(metrics, '\n')

if __name__ == "__main__":
    main()

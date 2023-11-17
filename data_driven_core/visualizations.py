'''
Helper functions for model visualization
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def feature_importances(model):
    '''
    display function to visualize relative importance of features in a classification model

    parameters:
        model: trained model object

    NO return (displays plot)
    '''

    # get feature names for y-axis
    encoder = model.steps[0][0]
    features = model.named_steps[encoder].get_feature_names_out()

    # calculate feature importances
    model_name = model.steps[-1][0]

    # get model coefficients if regression, gini if classifier
    if 'regression' in model_name \
    or 'ridge' in model_name \
    or 'lasso' in model_name:
        
        # remove feature metadata from name
        features_cleaned = [feature[feature.find('__')+2:].replace('Ref', '') \
                            for feature in features]

        data = model.named_steps[model_name].coef_
    else:
        features_cleaned = features
        data = model.named_steps[model_name].feature_importances_

    # create series of (coefficient or gini) values to plot
    pd.Series(data=data, index=features_cleaned).sort_values(key=abs).tail(20).plot(kind='barh')

    # plot feature name v. gini
    plt.ylabel('features')
    plt.xlabel('feature importance')
    plt.title(f'{model_name} feature importances')
    plt.show()


def predictions_v_true(y, y_pred):
    '''
    display function to vizualize predicted and true values for regression model

    parameters:
        y: (array) of target labels
        y_pred: (array) of predicted labels

    returns:
        None (displays scatter plot)
    '''

    # generate plot
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color='red', linewidth=2)
    plt.title('Predicted vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.show()

def confusion_matrix(model, y, y_pred):
    '''
    display function to visualize predicted and true values for all observations
    
    parameters:
        model: trained classifier model (used for name only)
        y: target vector
        y_pred: model output vector
    
    returns:
        None (displays confusion matrix plot)
    '''

    # generate confusion matrix and plot
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap='Blues')
    plt.title(f'{model.steps[2][0]} predictions v. actual positions')
    plt.show()

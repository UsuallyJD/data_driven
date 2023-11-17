'''
This file has functions that generate analysis metrics for a type of f1 model
Baseline metrics are returned alongisde those for the model
'''

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

def regression_metrics(model, X, y, six_fold_cv):
    '''
    calculator function that determines metrics for a REGRESSION model
    '''

    name = model.steps[-1][0]

    mae = -1 * cross_val_score(model,
                               X,
                               y.astype(int),
                               cv=six_fold_cv,
                               scoring='neg_mean_absolute_error').mean()
    
    r2 = cross_val_score(model,
                         X,
                         y.astype(int),
                         cv=six_fold_cv,
                         scoring='r2').mean()
    
    return [name, mae, r2]


def classifier_metrics(model, X, y, six_fold_cv):
    '''
    calculator function that determines metrics for a CLASSIFIER model
    '''

    name = model.steps[-1][0]
    # remove 'classifier' from name
    name = name[:name.find('classifier')]

    mae = -1 * cross_val_score(model, X, y, cv=six_fold_cv, scoring='neg_mean_absolute_error').mean()
    acc = cross_val_score(model, X, y, cv=six_fold_cv, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y, cv=six_fold_cv, scoring='f1_macro').mean()

    return [name, mae, acc, f1]


# def print_mae_grid(f1_data):
#     '''
#     Function docstring
#     '''

#     mae_grid = []

#     for feature in [col for col in f1_data.columns if f1_data[col].nunique() < 50]:
#         error = []

#         for element in f1_data[feature].value_counts().index:
#             data = f1_data.loc[f1_data[feature] == element]
#             error.append(mean_absolute_error([data['position'].mean()]*len(data['position']),
#                                              data['position']))

#         mae_grid.append([feature, sum(error)/len(error)])

#     print(mae_grid)

'''
These are the hyperparameter tuning functions for the random forest and
XG Boost model. They are excluded because their outputs are hard-coded into the
models because they take a while to run

The random forest function works and those values are used, the XG Boost one doesn't
because I'm not familiar with XG Boost HP tuning and it looks pretty dense. Room for
growth there, I'd like to do it correctly when I have the time
'''

from sklearn.model_selection import GridSearchCV

def rf_hp_tuning(model_rf):
    '''
    Function docstring
    '''

    rf_grid = {'randomforestclassifier__n_estimators': range(100, 301),
            'randomforestclassifier__criterion': ['gini', 'entropy'],
            'randomforestclassifier__max_depth': range(10, 101),
            'randomforestclassifier__min_samples_split': range(25, 76),
            'randomforestclassifier__min_samples_leaf': range(1, 25),
            'randomforestclassifier__max_features': ['sqrt', None]}

    clf_rf = GridSearchCV(model_rf, rf_grid, n_jobs=-1, cv=six_fold_cv, verbose=1)
    clf_rf.fit(X, y)
    print("Best set of hyperparameters: ", clf_rf.best_params_)
    print("Best score: ", clf_rf.best_score_)

    return None


def xg_hp_tuning(model_xg):
    '''
    Function docstring
    '''

    xg_grid = {'XGBClassifier__booster': ['gbtree', 'dart'],
            'XGBClassifier__gamma': (0, 10, 1),
            'XGBClassifier__max_depth': range(3, 11),
            'XGBClassifier__min_child_weight': [x / 100.0 for x in range(0, 101, 25)],
            'XGBClassifier__subsample': [x / 10.0 for x in range(5, 11, 1)],
            'XGBClassifier__tree_method': ['exact']}

    clf_xg = GridSearchCV(model_xg, xg_grid, n_jobs=-1, cv=six_fold_cv, verbose=1)
    clf_xg.fit(X, y_prime)
    print("Best set of hyperparameters: ", clf_xg.best_params_)
    print("Best score: ", clf_xg.best_score_)

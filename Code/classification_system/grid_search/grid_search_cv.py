from sklearn.model_selection import GridSearchCV

"""
grid_search_cv.py is used to perform a Grid Search cross-validation on a provided
models hyperparameters to select the optimal settings. It utilizes the GridSearchCV
function from the sklearn library.

@Author: Siôn Davies
Date: July 2020
"""


def grid_search(clf, parameters, X_train, y_train):
    """
    Performs a Grid Search on the given models hyperparameters.

    :param clf: The pipeline object containing the model to tune.
    :param parameters: The list of possible parameters to test for the given model.
    :param X_train: X training data.
    :param y_train: y training data.
    :return: The provided model whose parameters are now optimal for classification purposes.
    """
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, scoring='f1_weighted', cv=5)
    grid_search.fit(X_train, y_train)
    clf.set_params(**grid_search.best_params_)
    print('Parameters selected for model: ' + str(clf))
    return clf


"""
Hyperparameters for Linear SVC model:
- C: The Regularization parameter. The inverse of regularization strength. 
- max_iter: The maximum number of iterations to be run. 
- loss: Specifies loss function. 
"""
lin_svc_grid = [
    {
        'clf__C': [0.5, 1.0, 10.0, 100.0, 1000.0],
        'clf__max_iter': [10000, 20000, 30000],
        'clf__loss': ['hinge', 'squared_hinge']
    }
]

"""
Hyperparameters for Logistic Regression model:
- C: The Regularization parameter. The inverse of regularization strength. 
- max_iter: The maximum number of iterations to be run. 
- Solver: Algorithm to use in the optimization problem.
- Penalty: Specifies the norm used in penalization. 
"""
log_reg_grid = [
    {
        'clf__C': [0.5, 1.0, 10.0, 100.0, 1000.0],
        'clf__max_iter': [1000, 2000, 4000, 10000],
        'clf__solver': ['liblinear'],
        'clf__penalty': ['l1']
    },
    {
        'clf__C': [1, 10, 100, 1000],
        'clf__max_iter': [1000, 2000, 4000],
        'clf__solver': ['lbfgs', 'newton-cg'],
        'clf__penalty': ['l2']
    }
]

"""
Hyperparameters for Random Forest model:
- n_estimators: The number of Trees in the forest. 
- min_samples_split: The minimum number of samples required to split an internal node.
- min_samples_leaf: The minimum number of samples required to be at a leaf node. 
- bootstrap: boolean to decide if bootstrap samples should be used when building Trees. 
"""
rand_forest_grid = [
    {
        # 'clf__bootstrap': [True, False],
        'clf__n_estimators': [10, 100, 200],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
    }
]

smote_forest_grid = {

}



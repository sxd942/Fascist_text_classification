from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

"""

models.py contains the algorithms used to trained the supervised learning models.

The models include:
- Random Forests
- Linear SVC (Support Vector Classifier)
- Logistic Regression

The models hyperparameters are left as default, apart from Logistic Regression whose max_iter
parameter has been set to 2000. This is because if it is less than this value it will fail to 
converge if grid_search_tuning is set to 'False'.

The models optimal parameters are then selected if desired by selecting grid_search_tuning as 
'True' in classification_experiments.py via Grid Search tuning. 

@Author Siôn Davies
Date: July 2020

"""


def get_random_forest():
    """
    Gets the Random Forest model.

    :return: Random Forest model.
    """
    random_forest = RandomForestClassifier(random_state=42)
    return random_forest


def get_linear_svc():
    """
    Gets the Linear SVC model.

    :return: Linear SVC model.
    """
    svc = LinearSVC(random_state=42)
    return svc


def get_logistic_regression():
    """
    Gets the Logistic Regression model.

    :return: Logistic Regression model.
    """
    logistic_regression = LogisticRegression(random_state=42, max_iter=2000)
    return logistic_regression


def get_all_models():
    """
    Gets all models.

    :return: A list of all models.
    """
    rf = get_random_forest()
    svc = get_linear_svc()
    lr = get_logistic_regression()
    classifier_list = []
    classifier_list.extend(rf, svc, lr)
    return classifier_list


"""
Model constants which are used as params in classification_experiments.py
"""
svc = get_linear_svc()
log_reg = get_logistic_regression()
ran_forest = get_random_forest()

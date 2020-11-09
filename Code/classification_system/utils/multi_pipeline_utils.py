import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from utils.common_constants import StratifiedKFold
from utils.pipeline_utils import get_parameters
from grid_search.grid_search_cv import grid_search

"""
mutli_pipeline_utils.py contains utility functions for the multi-class classification process
that occurs in classification_experiments.py.

@Author: Siôn Davies
Date: August 2020
"""


def get_classes():
    """
    Getter for fascist vs. hate speech dataset.

    :return: dataset.
    """
    data = pd.read_csv('../Datasets/Multiclass/Hate_Fascist_Gold.csv')
    return data


def get_both_classes():
    """
    Getter for fascist vs. hate vs. both dataset.

    :return: dataset.
    """
    data = pd.read_csv('../Datasets/Multiclass/Hate_Fascist_Both.csv')
    return data


def get_both_synthetic():
    """
    Getter for synthetic fascist vs. hate dataset.

    :return: dataset.
    """
    data = pd.read_csv('../Datasets/Multiclass/Hate_Fascist_Synthetic.csv')
    return data


def get_multi_shuffle():
    """
    Getter for the Shuffled multi-class train and test datasets.

    :return: datasets.
    """
    shuffle_train = pd.read_csv('../Datasets/Multiclass/Multi_Shuffled_Train.csv')
    shuffle_test = pd.read_csv('../Datasets/Multiclass/Multi_Shuffled_Test.csv')
    return shuffle_train, shuffle_test


def set_input_target(data):
    """
    Setter for X and y categories.

    :param data: the dataset to divide.
    :return: X: input variable, y: output target variable.
    """
    X = data.Message_Post
    y = data.Numeric_Label
    return X, y


def multi_pipeline(feature, model, X, y, grid_search_tuning):
    """
    Makes the multi-classification Pipeline.

    :param feature: Feature extraction method.
    :param model: Classification algorithm.
    :param X: input variable.
    :param y: output target variable.
    :param grid_search_tuning: Boolean.
    :return: Multi-class classification Pipeline.
    """
    pipeline = Pipeline([('feature_transformer', feature), ('clf', model)])
    if grid_search_tuning:
        pipeline = grid_search(pipeline, get_parameters(model), X, y)
    return pipeline


def get_folds():
    """
    Getter for folds for cross-validation.

    :return: 5 folds.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cv


def get_scores(clf, input_X, target_y, folds):
    """
    Getter for corss-validation scores.

    :param clf: Model.
    :param input_X: input variable.
    :param target_y: output target variable.
    :param folds: Number of folds.
    :return: cross-validation scores.
    """
    scores = cross_val_score(clf, input_X, target_y, cv=folds)
    return scores


def get_prediction(clf, input_X, target_y, folds):
    """
    Getter for cross validation predictions.

    :param clf: Model.
    :param input_X: input variable.
    :param target_y: output target variable.
    :param folds: Number of folds.
    :return: classifier predictions.
    """
    prediction = cross_val_predict(clf, input_X, target_y, cv=folds)
    return prediction


def print_mean_accuracy(scores):
    """
    Takes the cross-validation scores and computes the mean score.

    :param scores: cross-validation scores.
    :return: the mean of these scores.
    """
    mean_score = np.mean(scores)
    return print("Mean accuracy score: " + "{:.2f}".format(mean_score))









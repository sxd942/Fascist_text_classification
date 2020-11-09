import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from feature_extraction.doc2vec_features import *
from feature_extraction.tfidf_features import *
from feature_extraction.word2vec_features import *
from models.models import *
from grid_search.grid_search_cv import *
from imblearn.under_sampling import RandomUnderSampler

"""
Utility functions for Classification pipeline in classification_experiments.py

@Author Siôn Davies
Date: July 2020
"""


def set_train_test(train_df, test_df):
    """
    Separates train/test data into X and y train/test data.

    :param train_df: Training data with input and output variables.
    :param test_df: Test data with input and output variables.
    :return: Train and test data split into input X and output y categories.
    """
    X_train = train_df.Message_Post
    X_test = test_df.Message_Post
    y_train = train_df.Numeric_Label
    y_test = test_df.Numeric_Label
    return X_train, X_test, y_train, y_test


def classification_pipeline(feature, classifier, X_train, y_train, X_test, grid_search_tuning):
    """
    Creates sklearn classification pipeline with feature vectorizer and model.

    :param feature: The feature vectorizer to transform the text data into numeric feature vectors.
    :param classifier: The learning model.
    :param X_train: The input training documents.
    :param y_train: The output training labels.
    :param X_test: The input test documents
    :param grid_search_tuning: if True, perform Grid Search cv hyperparameter tuning, else use default settings.
    :return: Classification model and predictions made on test documents.
    """
    model = Pipeline([('features', feature), ('clf', classifier)])
    if grid_search_tuning:
        _model = grid_search(model, get_parameters(classifier), X_train, y_train)
        _model.fit(X_train, y_train)
        y_prediction = _model.predict(X_test)
    else:
        _model = model.fit(X_train, y_train)
        y_prediction = _model.predict(X_test)
    return _model, y_prediction


def SMOTE_classification_pipeline(df, feature, classifier, grid_search_tuning):
    """
    Performs sklearn train/test split
    Performs SMOTE oversampling on imbalanced minority classes training data.
    Creates sklearn classification pipeline for synthetic data.

    :param df: The data frame containing
    :param feature: The feature vectorizer to transform the text data into numeric feature vectors.
    :param classifier: The learning model.
    :param grid_search_tuning: if True, perform Grid Search cv hyperparameter tuning, else use default settings.
    :return: Classification model, predictions made on test documents, and the test output labels.
    """
    feature_transformer = check_feature(feature)
    # Transforms X via selected feature vectors
    X = feature_transformer.fit_transform(df.Message_Post)
    y = df.Numeric_Label
    # Perform sklearn train test split on X and y.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    sm = SMOTE(random_state=2)
    # Perform SMOTE oversampling to balance the minority class (only on training data).
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    # Perform random under-sampling on non-fascist test samples to balance classes.
    X_test_res, y_test_res = under_sample(X_test, y_test)
    if grid_search_tuning:
        model = Pipeline([('clf', classifier)])
        model = grid_search(model, get_smote_parameters(classifier), X_train_res, y_train_res)
        model.fit(X_train_res, y_train_res)
        y_prediction = model.predict(X_test_res)
    else:
        model = Pipeline([('clf', classifier)])
        model = model.fit(X_train_res, y_train_res)
        y_prediction = model.predict(X_test_res)

    return model, y_prediction, y_test_res


def under_sample(X_test, y_test):
    """
    Function to perform under-sampling of majority class.
    Balances SMOTE test data.

    :param X_test: The input test documents.
    :param y_test: The target test labels.
    :return: X_test and y_test with balanced classes.
    """
    rus = RandomUnderSampler(random_state=42)
    X_test_res, y_test_res = rus.fit_resample(X_test, y_test)
    return X_test_res, y_test_res


def check_feature(feature):
    """
    Function to return an object or transformer form of feature vectorizer.

    :param feature: Feature vectorizer to be checked.
    :return: Object or transformer form of feature vectorizer.
    """
    if (feature == Tfidf_word_vec_features or feature == Tfidf_char_vec_features
       or feature == Doc2vec_features or feature == Word2Vec_features):
        return feature()
    else:
        return feature


def get_parameters(model):
    """
    Function to get the hyperparameters of a model to be passed to Grid Search cv.

    :param model: A given model to tune.
    :return: A combination of hyperparameters for the given model.
    """
    if model == svc:
        return lin_svc_grid
    elif model == log_reg:
        return log_reg_grid
    elif model == ran_forest:
        return rand_forest_grid


def get_smote_parameters(model):
    """
    Function to get the hyperparameters of a model to be passed to Grid Search cv.
    For SMOTE synthetic dataset.

    :param model: A given model to tune.
    :return: A combination of hyperparameters for the given model.
    """
    if model == svc:
        return lin_svc_grid
    elif model == log_reg:
        return log_reg_grid
    elif model == ran_forest:
        return smote_forest_grid


def get_gold_train_test():
    """
    Getter for Gold train/test data.

    :return: Gold train and test data.
    """
    gold_train = pd.read_csv('../Datasets/Gold/Gold_train_1.csv')
    gold_test = pd.read_csv('../Datasets/Gold/Gold_test_1.csv')
    return gold_train, gold_test


def get_gold_cross_val():
    gold_df = pd.read_csv('../Datasets/Gold/Gold_cleaned_1.csv')
    return gold_df


def get_shuffled_train_test():
    """
    Getter for Shuffled train/test data.

    :return: Shuffled train and test data.
    """
    shuffled_train = pd.read_csv('../Datasets/Shuffled/Shuffle_train_1.csv')
    shuffled_test = pd.read_csv('../Datasets/Shuffled/Shuffle_test_1.csv')
    return shuffled_train, shuffled_test


def get_SR_train_test():
    """
    Getter for SR train and test data.

    :return: SR train and test data.
    """
    SR_train = pd.read_csv('../Datasets/SR/SR_train.csv')
    SR_test = pd.read_csv('../Datasets/SR/SR_test.csv')
    return SR_train, SR_test


def get_synthetic_data():
    """
    Getter for Synthetic dataset.

    :return: Synthetic datatset.
    """
    synthetic_df = pd.read_csv('../Datasets/Synthetic/Synthetic_clean_1.csv')
    return synthetic_df




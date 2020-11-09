import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, \
    precision_recall_fscore_support

"""

classification_metrics.py contains all the methods that are used to gather and quantify
the quality of the predictions made by the trained models and utilizes the sklearn
metrics library. 

The methods evaluate and print: 
- Accuracy Scores 
- Confusion matrices
- Classification reports
- Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
- Precision, Recall and f1 scores.

@Author Siôn Davies
Date: July 2020

"""


def get_accuracy_score(y_test, prediction):
    """
    Gets accuracy score based on the models prediction of X_data.

    :param y_test: The correct labels for the test data.
    :param prediction: The prediction made by the classifier on unseen X_test_data.
    :returns: The models accuracy score.
    """
    acc_score = metrics.accuracy_score(y_test, prediction)
    return print('Accuracy score:\n %s' % acc_score + '\n')


def make_confusion_matrix(cm, string):
    """
    Makes and prints a confusion matrix for the selected classifier.

    :param cm: Confusion Matrix with predictions and y_test labels.
    :param string: String name of dataset that has been used to display.
    :return: Displays confusion matrix for classifier.
    """
    ax = plt.subplot()
    sns.heatmap(cm, annot=True,
           fmt='.2%', cmap='Blues', ax=ax)
    plt.title(string + 'Confusion Matrix (normalized)')
    ax.xaxis.set_ticklabels(['Non-Fascist', 'Fascist'])
    ax.yaxis.set_ticklabels(['Non-Fascist', 'Fascist'])
    plt.xticks(rotation=45)
    plt.show()


def get_confusion_matrix(y_test, prediction, string):
    """
    Gets the confusion matrix and passes it to the make_confusion_matrix function.
    Also normalises the matrix in the process.

    :param y_test: The correct labels for the test data.
    :param prediction: The prediction made by the classifier on unseen X_test_data.
    :param string: String name of dataset that has been used to display.
    """
    cm = confusion_matrix(y_test, prediction)
    norm_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('confusion matrix:\n %s' % cm)
    print('\n')
    make_confusion_matrix(norm_matrix, string)


def get_classification_report(y_test, prediction):
    """
    Makes and prints Classification Report for the selected classifier.

    :param y_test: The correct labels for the test data.
    :param prediction: The prediction made by the classifier on unseen X_test_data.
    :return: Prints a Classification Report for selected classifier.
    """
    class_report = metrics.classification_report(y_test, prediction)
    return print('Classification report:\n %s' % class_report)


def get_prec_rec_f1(y_test, prediction):
    """
    Gets and prints the Precision / Recall / F1 score for the selected classifier.

    :param y_test: The correct labels for the test data.
    :param prediction: The prediction made by the classifier on unseen X_test_data.
    :return: Prints the Precision / Recall / F1 score for the selected classifier.
    """
    score = metrics.precision_recall_fscore_support(y_test, prediction)
    return print('Precision, Recall, f1, support:\n'), print(score)


def get_roc_score(y_test, prediction):
    """
    Gets and print the ROC AUC score for the selected classifier.

    :param y_test: The correct labels for the test data.
    :param prediction: The prediction made by the classifier on unseen X_test_data.
    :return: Prints the ROC AUC score for the selected classifier.
    """
    score = metrics.roc_auc_score(y_test, prediction)
    return print('ROC AUC score:\n %s' % score)


def get_multi_class_report(target_y, target_pred):
    """
    Prints a classification report for the multi-class
    classification process.

    :param target_y: The correct labels for the test data.
    :param target_pred: The prediction made by the classifier on unseen X_test_data.
    :return: Classification report for multi-classes.
    """
    multi_class_report = classification_report(target_y, target_pred)
    return print('Classification report:\n %s' % multi_class_report)


def get_multi_matrix(target_y, target_pred, both, string):
    """
    Gets the confusion matrix and passes it to the make_confusion_matrix function.
    Also normalises the matrix in the process. Used for multi-class classification.

    :param string: The name of the dataset to print.
    :param both: if True print additional label 'both'.
    :param target_y: The correct labels for the test data.
    :param target_pred: The prediction made by the classifier on unseen X_test_data.
    """
    cm = confusion_matrix(target_y, target_pred)
    norm_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('confusion matrix:\n %s' % cm)
    print('\n')
    make_multi_confusion_matrix(norm_matrix, both, string)


def make_multi_confusion_matrix(cm, both, string):
    """
    Makes and prints a confusion matrix for the selected classifier.
    Used for multi-class classification.

    :param string: The name of the dataset to print.
    :param both: if True print additional label 'both'.
    :param cm: Confusion Matrix with predictions and y_test labels.
    :return: Displays confusion matrix for classifier.
    """
    ax = plt.subplot()
    sns.heatmap(cm, annot=True,
           fmt='.2%', cmap='Blues', ax=ax)
    plt.title(string + 'Confusion Matrix (normalized)')
    if both:
        ax.xaxis.set_ticklabels(['Neither', 'Fascist', 'Hate', 'Both'])
        ax.yaxis.set_ticklabels(['Neither', 'Fascist', 'Hate', 'Both'])
    else:
        ax.xaxis.set_ticklabels(['Neither', 'Fascist', 'Hate'])
        ax.yaxis.set_ticklabels(['Neither', 'Fascist', 'Hate'])
    plt.xticks(rotation=45)
    plt.show()


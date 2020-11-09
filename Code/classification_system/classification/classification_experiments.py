from classification.classification_metrics import *
from feature_extraction.feature_constants import *
from utils.pipeline_utils import *
from utils.common_constants import *
from preprocessing.preprocess import stem
from utils.multi_pipeline_utils import *
"""
@Author: Siôn Davies
Date: July 2020
"""

"""
make_classification() is used to perform the binary classification experiments in the 
project - Fascist vs Non.Fascist. 
It allows the user to select their desired model and desired feature extraction technique, 
it then builds four models based on these selections, one for each dataset (Gold,
Shuffled, SR and Synthetic). It then prints out performance metrics to display
how well each classifier has performed.

make_classification() has 3 parameters: make_classification(feature, model, grid_search_tuning)

Parameter 1: feature 
The feature extraction technique to use. Options are as follows:
 a.) Tf-idf Word-grams -> insert param: 'tfidf_word'
 b.) Tf-idf char-grams -> insert param: 'tfidf_char'
 c.) Word2Vec (mean word embeddings) -> insert param: 'word2vec'
 d.) Doc2Vec (paragraph embeddings) -> insert param: 'doc2vec'

Parameter 2: Model
The algorithm to use. Options are as follows: 
a.) Linear-SVC -> insert param: 'svc'
b.) Logistic Regression -> insert param: 'log_reg'
c.) Random Forest -> insert param: 'ran_forest'

Parameter 3: grid_search_tuning
Boolean value to decide whether or not to perform Grid Search cv to tune each models parameters
to optimal settings:
a.) 'True'
b.) 'False'
Please be warned, if selected as True, the program will take considerable time to terminate due to the number 
of possible parameter combinations that must be trialled. 
If selected as False, the default settings for the models parameters will be used instead.
"""


def make_classification(feature, model, grid_search_tuning):

    print("\nLoading models.... (time dependant upon feature extraction method "
          "and parameter tuning selection)\n")

    # 1. Get training and test data (stem and remove stopwords)...
    print('Gathering data....\n')
    print('Pre-processing data (stemming text and removing english stop-words)...\n')
    gold_df = get_gold_cross_val()
    gold_df.Message_Post = gold_df.Message_Post.apply(stem)
    gold_input_X, gold_target_y = set_input_target(gold_df)

    shuffled_train, shuffled_test = get_shuffled_train_test()
    shuffled_train.Message_Post = shuffled_train.Message_Post.apply(stem)
    shuffled_test.Message_Post = shuffled_test.Message_Post.apply(stem)

    SR_train, SR_test = get_SR_train_test()
    SR_train.Message_Post = SR_train.Message_Post.apply(stem)
    SR_test.Message_Post = SR_test.Message_Post.apply(stem)

    synthetic_df = get_synthetic_data()
    synthetic_df.Message_Post = synthetic_df.Message_Post.apply(stem)

    # 2. Get X and y train / test sets for non-synthetic datasets.
    shuffled_X_train, shuffled_X_test, shuffled_y_train, shuffled_y_test = set_train_test(shuffled_train,
                                                                                          shuffled_test)
    SR_X_train, SR_X_test, SR_y_train, SR_y_test = set_train_test(SR_train, SR_test)
    print('Test and Training data gathered.\n')

    # 3. Create classification Pipelines for synthetic and non-synthetic data.
    print(SHUFFLED + 'training classifier.\n')
    shuffled_clf, shuffled_predict = classification_pipeline(check_feature(feature), model, shuffled_X_train,
                                                             shuffled_y_train, shuffled_X_test
                                                             , grid_search_tuning=False)
    print(SHUFFLED + 'classifier trained.\n')

    print(SR + 'training classifier.\n')
    SR_clf, SR_predict = classification_pipeline(check_feature(feature), model, SR_X_train,
                                                 SR_y_train, SR_X_test, grid_search_tuning=False)
    print(SR + 'classifier trained.\n')

    print('SMOTE oversampling beginning...Training synthetic classifier (...time) \n')
    synthetic_clf, synthetic_predict, synthetic_y_test = SMOTE_classification_pipeline(synthetic_df,
                                                                    feature, model, grid_search_tuning=False)
    print(SYNTHETIC + 'classifier trained.\n')

    if grid_search_tuning:
        print('Performing Grid search tuning to optimize Gold model parameters. (...time)\n')
    else:
        print(GOLD + 'training classifier via cross validation.\n')
    gold_clf = multi_pipeline(feature, model, gold_input_X, gold_target_y, grid_search_tuning)
    print("Establishing n folds for cross-validation.\n")
    folds = get_folds()
    print("Getting mean Gold classifier scores.\n")
    gold_scores = get_scores(gold_clf, gold_input_X, gold_target_y, folds)
    print("Getting Gold classifier predictions.\n")
    gold_target_pred = get_prediction(gold_clf, gold_input_X, gold_target_y, folds)
    print(GOLD + 'classifier trained.\n')

    # 4. Gather evaluation metrics.
    print('****************************************\n')
    print('********* Evaluation Report ************')
    # Accuracy scores.
    print('Accuracy Scores...\n')
    print('Gold dataset (Mean score):'), print_mean_accuracy(gold_scores), print('\n')
    print('Shuffled dataset:'), get_accuracy_score(shuffled_y_test, shuffled_predict)
    print('Synonym Replacement dataset:'), get_accuracy_score(SR_y_test, SR_predict)
    print('SMOTE synthetic dataset:'), get_accuracy_score(synthetic_y_test, synthetic_predict)

    print('****************************************\n')
    # Confusion Matrices.
    print('Confusion Matrices...\n')
    print('Gold dataset:'), get_confusion_matrix(gold_target_y, gold_target_pred, GOLD), print('\n')
    print('Shuffled dataset:'), get_confusion_matrix(shuffled_y_test, shuffled_predict, SHUFFLED), print('\n')
    print('Synonym Replacement dataset:'), get_confusion_matrix(SR_y_test, SR_predict, SR), print('\n')
    print('SMOTE synthetic dataset:'), get_confusion_matrix(synthetic_y_test, synthetic_predict, SYNTHETIC)

    print('****************************************\n')
    # Classification Reports.
    print('Classification Reports...\n')
    print('Gold dataset:'), get_multi_class_report(gold_target_y, gold_target_pred)
    print('Shuffled dataset:'), get_classification_report(shuffled_y_test, shuffled_predict)
    print('Synonym Replacement dataset:'), get_classification_report(SR_y_test, SR_predict)
    print('SMOTE synthetic dataset:'), get_classification_report(synthetic_y_test, synthetic_predict)

    print('****************************************\n')
    # ROC AUC Scores.
    print('ROC AUC scores...\n')
    print('Gold dataset:'), get_roc_score(gold_target_y, gold_target_pred), print('\n')
    print('Shuffled dataset:'), get_roc_score(shuffled_y_test, shuffled_predict), print('\n')
    print('Synonym Replacement dataset:'), get_roc_score(SR_y_test, SR_predict), print('\n')
    print('SMOTE synthetic dataset:'), get_roc_score(synthetic_y_test, synthetic_predict)
    print('\n****************************************\n')
    print('******* End of Report **********')
    return


# make_classification(tfidf_word, log_reg, False)

"""
make_multi_classification() is used to perform the multi-class classification experiments in 
the project: Fascist vs. Hate and Fascist vs. Hate vs. Both

As only the Gold, Shuffled and Synthetic datasets were used during this investigations, this method will only
generate results for these datasets. Additionally, as the Gold dataset was only used when combined with 
the 'Both' class, if the parameter 'both' is set to True, it will only generate results for the Gold dataset.

make_classification() has 4 parameters: make_classification(feature, model, grid_search_tuning, both)

Parameter 1: feature 
The feature extraction technique to use. Options are as follows:
 a.) Tf-idf Word-grams -> insert param: 'tfidf_word'
 b.) Tf-idf char-grams -> insert param: 'tfidf_char'
 c.) Word2Vec (mean word embeddings) -> insert param: 'word2vec'
 d.) Doc2Vec (paragraph embeddings) -> insert param: 'doc2vec'

Parameter 2: Model
The model to use. Options are as follows: 
a.) Linear-SVC -> insert param: 'svc'
b.) Logistic Regression -> insert param: 'log_reg'
c.) Random Forest -> insert param: 'ran_forest'

Parameter 3: grid_search_tuning
Boolean value to decide whether or not to perform Grid Search cv to tune each models parameters
to optimal settings:
a.) 'True'
b.) 'False'
Please be warned, if selected as True, the program will take considerable time to terminate due to the number 
of possible parameter combinations that must be trialled. 
If selected as False, the default settings for the models parameters will be used instead. 

Parameter 4: both
Boolean value to decide whether to include the additional class 'both' in the classification process, where the class
both indicates documnets that include attributes defined as both fascist and hate speech. This option only applies
to the Gold dataset:
a.) 'True'
b.) 'False'
Please note - the intentions behind this parameter were not incorporated into the final project.
Therefore, to recreate the experiments from the research, this parameter should be set to 'False'.
"""


def make_multi_classification(feature, model, grid_search_tuning, both):
    # 1.) Get the Data
    print("Loading data...\n")
    if both:
        data = get_both_classes()
    else:
        data = get_classes()
    shuffled_train, shuffled_test = get_multi_shuffle()
    # 2.) Last phase of pre-processing -> remove stop-words and stem...
    print("Pre-processing text data (stemming and removing stop-words)...\n")
    data.Message_Post = data.Message_Post.apply(stem)
    shuffled_train.Message_Post = shuffled_train.Message_Post.apply(stem)
    shuffled_test.Message_Post = shuffled_test.Message_Post.apply(stem)
    # 3.) Split into X and y
    print("Getting input and target variables...\n")
    input_X, target_y = set_input_target(data)
    shuffled_X_train, shuffled_X_test, shuffled_y_train, shuffled_y_test = set_train_test(shuffled_train,
                                                                                          shuffled_test)
    # 4.) Create SMOTE classification pipeline
    print("Building classification pipeline.\n")
    if both is False:
        print('SMOTE oversampling beginning...Training synthetic classifier (...time) \n')
        synthetic_df = get_both_synthetic()
        synthetic_clf, synthetic_predict, synthetic_y_test = SMOTE_classification_pipeline(synthetic_df, feature, model,
                                                                                           grid_search_tuning=False)
        print(SYNTHETIC + 'classifier trained.\n')
    # 4.25) Create Shuffle classification pipeline
        print(SHUFFLED + 'training classifier.\n')
        shuffled_clf, shuffled_predict = classification_pipeline(check_feature(feature), model, shuffled_X_train,
                                                                 shuffled_y_train, shuffled_X_test,
                                                                 grid_search_tuning=False)
    print(SHUFFLED + 'classifier trained.\n')
    # 4.5) Perform Grid Search optimization for hyper-parameters if True.
    if grid_search_tuning:
        print("Performing Grid search tuning to optimize model parameters. (...time)\n")
    clf = multi_pipeline(feature, model, input_X, target_y, grid_search_tuning)
    # 5.) Get folds
    print("Establishing n folds for cross-validation.\n")
    folds = get_folds()
    # 6.) Get cross-validation scores
    print("Getting mean models scores.\n")
    scores = get_scores(clf, input_X, target_y, folds)
    # 7.) Get classifier prediction
    print("Getting predictions.\n")
    target_pred = get_prediction(clf, input_X, target_y, folds)
    print('****************************************\n')
    # Now provide evaluation report.
    print('********* Evaluation Report ************\n')
    # 8.) Print mean scores
    print('Accuracy Scores...\n')
    print('Gold dataset (Mean score):'), print_mean_accuracy(scores), print('\n')
    if both is False:
        print('Shuffled dataset:'), get_accuracy_score(shuffled_y_test, shuffled_predict), print('\n')
        print('SMOTE synthetic dataset:')
        get_accuracy_score(synthetic_y_test, synthetic_predict)
    print('****************************************\n')
    # 9.) Get classification reports
    print('Classification Reports...\n')
    print('Gold dataset:'), get_multi_class_report(target_y, target_pred)
    if both is False:
        print('Shuffled dataset:'), get_multi_class_report(shuffled_y_test, shuffled_predict)
        print('SMOTE synthetic dataset:'), get_multi_class_report(synthetic_y_test, synthetic_predict)
    # 10.) Get Confusion metrics
    print('****************************************\n')
    print('Confusion Matrices...\n')
    print('Gold dataset:'), get_multi_matrix(target_y, target_pred, both, GOLD)
    if both is False:
        print('Shuffled dataset:'), get_multi_matrix(shuffled_y_test, shuffled_predict, both, SHUFFLED)
        print('Synthetic dataset:')
        get_multi_matrix(synthetic_y_test, synthetic_predict, both, SYNTHETIC)
    print('\n****************************************\n')
    print('******* End of Report **********')
    return


# make_multi_classification(tfidf_word, svc, grid_search_tuning=False, both=False)

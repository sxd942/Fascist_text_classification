import matplotlib.pyplot as plt
import numpy as np

"""
evaluation.py was used to create the bar charts used in the evaluation of the
algorithms and feature extraction techniques for the binary classification
experiments.

@Author: Siôn Davies
Date: 15/08/2020
"""

# First we begin with a bar chart for the algorithm comparison.

models = ['Linear SVC', 'Logistic Regression', 'Random Forest']
precision = [0.86, 0.83, 0.89]
recall = [0.83, 0.85, 0.68]
F1 = [0.83, 0.77, 0.72]
xPos = np.arange(len(models))

plt.title('Mean Algorithm Performance (Fascist document classification)')
plt.xticks(xPos, models)
plt.bar(xPos-0.2, F1, width=0.17, color='lightslategrey', label='F1')
plt.bar(xPos+0.0, precision, width=0.17, color='thistle', label='Precision')
plt.bar(xPos+0.2, recall, width=0.17, label='Recall')
plt.ylim(ymin=0.5)
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper center', fancybox=True)
plt.show()

# Now we do the same to evaluate the feature extraction methods...

features = ['TF-IDF Word n-grams', 'TF-IDF Char n-grams', 'Word embeddings', 'Paragraph vector']
feature_Precision = [0.94, 0.89, 0.82, 0.77]
feature_Recall = [0.74, 0.82, 0.84, 0.75]
feature_F1 = [0.80, 0.83, 0.81, 0.75]
feature_xPos = np.arange(len(features))

plt.title('Mean Feature Performance (Fascist document classification)')
plt.xticks(feature_xPos, features, rotation=20)
plt.bar(feature_xPos-0.2, feature_F1, width=0.17, color='lightslategrey', label='F1')
plt.bar(feature_xPos+0.0, feature_Precision, width=0.17, color='thistle', label='Precision')
plt.bar(feature_xPos+0.2, feature_Recall, width=0.17, label='Recall')
plt.ylim(ymin=0.5)
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper center', fancybox=True)
plt.show()

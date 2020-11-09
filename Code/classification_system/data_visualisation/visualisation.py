import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
"""
visulisation.py was used to create visual representations of
data used in various datasets in the project.
The images generated here have been included in the
'Dataset Selection' chapter of the project report.
To run this program the paths for the various csv files
will need to be in the correct path. 

@Author: Siôn Davies
@Version 01/07/20
"""

class DataVisualisation:
    # Read in the various datasets to be displayed...
    silver = pd.read_csv('../Datasets/dataset_utils/Silver.csv')
    gold = pd.read_csv('../Datasets/dataset_utils/Gold.csv')
    synthetic = pd.read_csv('../Datasets/dataset_utils/Synthetic.csv')

    # First we create a Strip Plot comparing the character counts of documents in the Iron March and Reddit datasets.
    silver.rename(columns={'Character_Length': 'Character Count'}, inplace=True)
    sns.stripplot(x='Label', y='Character Count', data=silver)
    plt.show()

    # Second, we create a pie chart displaying the fascist category breakdowns in the Gold fascist samples.
    categories = gold.drop(
        ['Source Dataset', 'Message_Post', 'Label', 'Fascist_Speech', 'Forum', 'String_Length', 'Language_ID',
         'Character_Length'], axis=1)

    categories.isnull().sum()
    categories.dropna(inplace=True)
    categories.isnull().sum()

    pieChart = categories.groupby(categories["Category"])["Index"].count()
    axis('equal')
    pie(pieChart, labels=pieChart.index, autopct='%1.1f%%', textprops={'fontsize': 14}, shadow=True, radius=2)
    plt.show()

    # Third, we create a Facet Grid, to contrast the sample quantities between the gold and silver datasets.
    silverGrid = sns.FacetGrid(silver, hue='Label', palette='coolwarm')
    silverGrid = silverGrid.map(plt.hist, 'Label', bins=5, alpha=0.7)
    silverGrid.axes[0, 0].set_xlabel('Silver Dataset')
    silverGrid.axes[0, 0].set_ylabel('Number of samples')
    plt.show()

    goldGrid = sns.FacetGrid(gold, hue='Label', palette='coolwarm')
    goldGrid = goldGrid.map(plt.hist, 'Label', bins=5, alpha=0.7)
    goldGrid.axes[0, 0].set_xlabel('Gold Dataset')
    goldGrid.axes[0, 0].set_ylabel('Number of samples')
    plt.show()

    # Finally, we demonstrate the power of SMOTE to perform oversampling on minority classes.

    def converter(column):
        if column == 'Yes':
            return 1
        else:
            return 0

    synthetic['Cluster'] = synthetic['Fascist_Speech'].apply(converter)

    tfidf_vec = TfidfVectorizer(stop_words='english')
    X = tfidf_vec.fit_transform(synthetic['Message_Post'])
    X_train, X_test, y_train, y_test = train_test_split(X, synthetic['Cluster'], random_state=2)
    sm = SMOTE(random_state=2)

    # First we visualise the fascist training samples there were originally...
    sns.set_style('darkgrid')
    sns.countplot(y_train, data=synthetic)
    plt.show()

    # Now we perform SMOTE oversampling to the minority class...
    X_res, y_res = sm.fit_sample(X_train, y_train)
    sns.set_style('darkgrid')
    sns.countplot(y_res, data=synthetic)
    plt.show()



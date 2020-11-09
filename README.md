ReadME file for Siôn Davies: Machine Learning for Fascist Text Classification in Social Networks.

REPOSITORY OVERVIEW:


The folder ‘Code’ contains two directories:

The first, ‘dataset_creation’, contains four IPYNB files that were written in a Jupyter Notebook and relate to the creation of the csv files for the original datasets.

The second, ‘classification_system’, contains the Python scripts that comprise the classification test system architecture that was detailed in chapter 6.3.4. All code contained in these scripts was written in Python version 3.8.

Contents:

- Sections 1 - 3: describe how to extract and run the code.
- Section 4: Lists code dependencies that must be in place in order for the code to run.
- Section 5: Lists code references that were used in this research.



1. EXTRACT THE DIRECTORY

To recreate the classification experiments that were performed in this research, download the directory from the repository as a Zip file and extract the directory ‘classification_system’ to your desired location. Inside your Python IDE of preference, open this directory and navigate to the package titled ‘classification’. Inside this package you will see the script named classification_experiments.py, select this script and use the instructions listed in section 2 to run the code.



2. CODE INSTRUCTIONS


To run the binary classification experiments...

i.) Uncomment the following method call: make_classification(feature, model, grid_search_tuning)
ii.) Replace the parameters with your chosen preferences and run the script.

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

Boolean value to decide whether or not to perform Grid Search cv to tune models parameters:

a.) 'True'

b.) 'False'

Please be warned, if selected as True, the program will take considerable time to terminate due to the number
of possible parameter combinations that must be trialled.
If selected as False, the default settings for the models parameters will be used instead.


To run the multiclass classification experiments...

i.) Uncomment the following method call: make_multi_classification(feature, model, grid_search_tuning, both).
ii.) Replace the parameters with your chosen preferences and run the script.

Parameters 1-3 are the same as above detailed in 2.A

Parameter 4: both

Boolean value to decide whether to include the additional class 'both' in the classification process, where the class
both indicates documents that include attributes defined as both fascist and hate speech. This option only applies
to the Gold dataset:

a.) 'True'

b.) 'False'

* Please note - the intentions behind this parameter were not incorporated into the final project. Therefore, to recreate
the experiments from the research, this parameter should be set to 'False'.



3. ADDITIONAL SCRIPTS

Additional scripts that can be run from inside the ‘classification_system’ directory include:

i.) The package ‘data_visualisation’ contains four Python scripts, each can be run individually to recreate the visual figures created for the project.

ii.) The package ‘classification’ contains a script titled model_analysis.py which can be run to view the results of the Tweet analysis.  



4. CODE DEPENDENCIES

Python version 3.8

i.) The following Python modules will need to be installed: numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, imbalanced-learn, genism, tqdm, inflect, string, re, wordcloud.


ii.) PLEASE NOTE: in order for the classification experiments to execute, the pre-trained word vectors that were used for the word embedding features will have to be downloaded onto your system. This happens automatically via the Gensim downloader API when either of make_classification() or make_multi_classification() is first run. This requires 1662.8MB of memory and takes several minutes to download.
More information about the word vectors can be found at: https://code.google.com/archive/p/word2vec/ under the section: "Pre-trained word and phrase vectors".



5. CODE REFERENCES

Below we reference resources which were utilized for coding purposes (these are also referenced in the respective scripts). All other code in the files is original.

(A) was used to perform the sentence reordering augmentation.

(B) was used to perform the synonym replacement augmentation.

(C) was adapted to create a Paragraph vector Doc2vec class (section: ‘Creating vector space model’).

(D) was adapted to create a mean word embedding Word2vec class (slide 18).

(E) contains reference to the pretrained word vectors that were used in this research.



(A) Tjihero, N., (2018). Data Augmentation For Text Data: Obtain More Data Faster. [online] Towards Data Science. Available at: <https://towardsdatascience.com/data-augmentation-for-text-data-obtain-more-data-faster-525f7957acc9> [Accessed 4 July 2020].

(B) Wei, J. and Zou, K., (2019). Jasonwei20/Eda_Nlp. [online] GitHub. Available at: <https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py> [Accessed 6 July 2020].

(C) Nag, A., (2019). A Text Classification Approach Using Vector Space Modelling(Doc2vec) & PCA. [online] Medium. Available at: <https://medium.com/swlh/a-text-classification-approach-using-vector-space-modelling-doc2vec-pca-74fb6fd73760> [Accessed 3 July 2020].

(D) Tyson, N., (2016). Word Embedding Models & Support Vector Machines For Text Classification. [online] Slideshare.net. Available at: <https://www.slideshare.net/bokononisms/word-embedding-models-support-vector-machines-for-text-classification> [Accessed 1 July 2020].

(E) code.google.com. (2013). Word2vec (Pre-Trained Word and Phrase Vectors). [online] Available at: <https://code.google.com/archive/p/word2vec/> [Accessed 7 July 2020].

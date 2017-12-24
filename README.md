# Restaurant-Review-Classification-using-NLTK-and-Naive-Bayes
This repository contains codebase and datafiles for the project titled "Restaurant Review Classification using NLTK and Naive Bayes


Steps :

1. The datset for this project is the Amazon Fine Food Review dataset available in Kaggle.

2. A subset of the dataset was taken for brevity. The entire dataset and the subset dataset are both included in the repository

3. In this project, the Natural Language ToolKit (NLTK) toolkit have been used for basic data preprocessing of Restaurant Reviews like special character removal,
creation of corpus of words and for stemming the words.

4. The code is written in a very modular format leaving scope for scalabilty in future.

   4.1 class Restaurant_Reviews : This class is the basic template of the project
   
   4.2 def __init__(self,dataset) : constructor
   
   4.3 def clean_data(self) : Method to clean the data and basic data preprocesing
   
   4.3 def bag_of_words(self,corpus): Method that buils up the bag of words model
   
   4.4 def train_test_split(self,X,y): Method that divides the data in Train and Test data
   
   4.5 def fit_classifier(self,X_train,y_train): Method that fits the Training data with a Naive Bayes classifier
   
   4.6 def test_classifier(self,classifier,X_test,y_test): Method that leverages the learned classifier object to predict the 
                                                           o/p for the test data and plot the confusion matrix and calculate the
                                                           Accuracy,Precison,Recall & F1score of the model


5. 
       ******Model Statistics ******
       
       Accuracy of Model =  0.73
       
       Precison of Model =  0.567010309278
       
       Recall of Model   =  0.820895522388
       
       F1score of Model  =  0.670731707317
       
        ****************************

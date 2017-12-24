# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:21:17 2017

@author: Kalyan
"""

#########################################################
#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#########################################################
#Defining the Class and its methods 

class Restaurant_Review(object):
    
    #Defining the member variable accessible to all classes
    corpus=[]
    X=None
    y=None
    X_train=None
    X_test=None
    y_train=None
    y_test=None
    classifier=None
    
    #Defining the constructor
    def __init__(self,dataset):
        
        self.dataset=dataset
        
    
    #Defining the clean data method    
    def clean_data(self):
        import re
        import nltk
        nltk.download('stopwords') #downloading the nltk stopwords directory
        from nltk.corpus import stopwords #Importing stopwords class 
        from nltk.stem.porter import PorterStemmer #Importing PorterStemmer class
        
        
        for i in range(0,1000):
            review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
            review=review.lower()
            review=review.split()
            ps=PorterStemmer()
            review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review=' '.join(review)
            self.corpus.append(review)
            return corpus
      
    #Defining the bag_of_words method    
    def bag_of_words(self,corpus):
        from sklearn.feature_extraction.text import  CountVectorizer
        cv=CountVectorizer(max_features=1500)
        X=cv.fit_transform(corpus).toarray()
        y=dataset.iloc[:,1].values
        return X,y 
        
        
    #Defining the Train/Test split method
    def train_test_split(self,X,y):
        from sklearn.cross_validation import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
        return X_train,X_test,y_train,y_test
        
    #Fitting the Naive Bayes Classifier to the Train data
    def fit_classifier(self,X_train,y_train):
        from sklearn.naive_bayes import GaussianNB
        classifier=GaussianNB()
        classifier.fit(X_train,y_train)
        return classifier
    
    #Fitting the learned model to the Test data and plotting the Confusing Matrix
    def test_classifier(self,classifier,X_test,y_test):
        y_pred=classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(y_test,y_pred)
        TP=cm[0][0]
        TN=cm[1][1]
        FP=cm[0][1]
        FN=cm[1][0]
        Accuracy=(TP+TN)/(TP+TN+FP+FN)
        Precison=TP/(TP+FP)
        Recall=TP/(TP+FN)
        F1score=(2*Precison*Recall)/(Precison+Recall)
        print ("******Model successfully tested******")
        print ("******Model Statistics ******")
        print ("Accuracy of Model = ",Accuracy)
        print ("Precison of Model = ",Precison)
        print ("Recall of Model   = ",Recall)
        print ("F1score of Model  = ",F1score)
        print ("****************************")
#########################################################        
        
#Importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Creating and instance of the class Restaurant_Review()
rr=Restaurant_Review(dataset)
#Getting the corpus from the clean_data method()
corpus=rr.clean_data()    
#Getting X,y from the bag_of_words method()
X,y=rr.bag_of_words(corpus)
#Getting X_train,X_test,y_train,y_test from the train_test_split() method
X_train,X_test,y_train,y_test=rr.train_test_split(X,y)
#Getting classifier object from the fit_classifier() method
classifier=rr.fit_classifier(X_train,y_train)
#Displaying the model statistics using the test_classifier() method
rr.test_classifier(classifier,X_test,y_test)
    
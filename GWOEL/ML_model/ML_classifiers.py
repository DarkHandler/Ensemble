import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier

import joblib #to save and load model

class ModuleEL: #singleton

    def __init__(self):
        self.data = pd.read_csv("metrics_results.txt", sep=",", index_col=False)
        self.X = self.data.drop(columns=['method'])
        self.y = self.data['method']
        #print(self.data.shape)

    def generateELModel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)

        # ----------- ZONA DE CLASSIFICACION ----------------
        modelDTC = DecisionTreeClassifier()#.fit(X_train, y_train)
        modelSVM = SVC(kernel='poly', probability=True)#.fit(X_train, y_train) #options= linear, poly, rbf, sigmoid
        modelGNB = GaussianNB()#.fit(X_train, y_train)
        modelRFC = RandomForestClassifier()#.fit(X_train, y_train)

        #--------------- ENSEMBLE LEARNING ZONE ------------------
        evc = VotingClassifier(estimators=[
            ('modelSVM', modelSVM),
            ('modelRFC', modelRFC),
            ('modelDTC', modelDTC),
            ('modelGNB', modelGNB)
            ], voting = 'soft') #uso soft para una mejor precision

        modelEL = evc.fit(X_train, y_train)

        predictionModelEL = modelEL.predict(X_test)

        #print(classification_report(y_test, predictionModelEL))

        #------- ZONA DE GUARDADO DE MODELOS ------------
        joblib.dump(modelEL, 'modelEL.joblib') #guardar Ensemble learning model

        return modelEL

    def loadELModel(self):
        modelEL = joblib.load('modelEL.joblib') #cargar el Ensemble Learning model
        return modelEL
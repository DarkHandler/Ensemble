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

from os.path import exists

import joblib #to save and load model

class ModuleEL:

    def __init__(self, dataFileName, modelName):
        self.data = pd.read_csv(dataFileName, sep=",", index_col=False)
        self.X = self.data.drop(columns=['method'])
        self.y = self.data['method']
        self.modelName = modelName

    def getELModel(self):
        file_exists = exists(self.modelName + '.joblib')
        if file_exists: #if the file exists
            return self.loadELModel()
        else:
            return self.generateELModel()


    def generateELModel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)

        # ----------- ZONA DE CREACION DE MODELOS BASE ----------------
        modelDTC = DecisionTreeClassifier()
        modelSVM = SVC(kernel='poly', probability=True) #options= linear, poly, rbf, sigmoid
        modelGNB = GaussianNB()
        modelRFC = RandomForestClassifier()

        #--------------- ENSEMBLE LEARNING ZONE ------------------
        evc = VotingClassifier(estimators=[
            ('modelSVM', modelSVM),
            ('modelRFC', modelRFC),
            ('modelDTC', modelDTC),
            ('modelGNB', modelGNB)
            ], voting = 'soft') #uso soft para una mejor precision y -1 para todos los procesadores

        #-Los labels deben tener como minimo dos clases, pero al ser 90% explotation, no se me esta generando y viendo ninguna fase de exploracion
        #labels = np.unique(y_train) ##---> aqui puede verse el error cuando probamos esta experimentacion 
        #print("NUMERO DE LABELS--: ", labels)
        
        modelEL = evc.fit(X_train, y_train)

        predictionModelEL = modelEL.predict(X_test)

        #print(classification_report(y_test, predictionModelEL))

        #------- ZONA DE GUARDADO DE MODELOS ------------
        joblib.dump(modelEL, self.modelName + '.joblib') #guardar Ensemble learning model

        return modelEL

    def loadELModel(self):
        modelEL = joblib.load(self.modelName + '.joblib') #cargar el Ensemble Learning model
        return modelEL
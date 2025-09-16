import os
from sklearn.utils import shuffle
import sys
sys.path.append('../')
from common import load_on_RAM, save
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

class ModelManager:
    def __init__(self, model, preprocess):
        """
        model must have "fit" function, defined like scikit-learn fit, and a attribute for the name
        """
        self.model = model
        self.preprocess = preprocess
    
    def train_model(self, X, y, save_path):
        X_train, y_train= self.preprocess_sets(X,y)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        print("(num_samples, num_features)")
        print(f"Train dataset : {X_train.shape}")

        self.model.fit(X_train, y_train)

        self.save_model(save_path)
    
    def test_model(self, X, y):
        X_test, y_test = self.preprocess_sets(X,y)

        y_pred = self.model.predict(X_test)

        print(classification_report(y_test, y_pred))

        print("Accuray score : " + str(accuracy_score(y_test, y_pred)))

    def preprocess_sets(self, X, y):
        # pre-process latents
        X_new, y_new = self.preprocess(X, y)
        del X

        y_new = np.array(y_new)
        X_new = np.array(X_new)
        return X_new,y_new

    def save_model(self, save_path):
        print(self.model.name + "will be saved into "+ save_path + " as "+ self.model.name)
        print(self.model)
        model_path = save(self.model, save_path, self.model.name)

        
        
    
        

class RandomForest(RandomForestClassifier):
    def __init__(self, name=None, **params):
        super().__init__(**params)
        if name is None:
            name = f"RF_{params['n_estimators']}estimators"
        self.name = name
    def __str__(self):
        return super().__str__()

class GridSearch(GridSearchCV):
    def __init__(self, estimator, name,params):
        super().__init__(estimator=estimator, param_grid=params, n_jobs=-1)
        if name is None:
            name = f"grid_search"
        self.name = name
    def __str__(self):
        return super().__str__()

class RandomizedSearch(RandomizedSearchCV):
    def __init__(self, estimator, name, params, n_iter=70, random_state=42):
        super().__init__(
            estimator=estimator, 
            param_distributions=params, 
            n_iter=n_iter,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        if name is None:
            name = f"randomized_search_{n_iter}iter"
        self.name = name
    def __str__(self):
        
        return str(self.best_estimator_)

class SVM(SVC):
    def __init__(self, name=None, **params):
        super().__init__(**params)
        if name is None:
            name = f"SVM_{params.get('kernel', 'rbf')}"
        self.name = name
    def __str__(self):
        return super().__str__()
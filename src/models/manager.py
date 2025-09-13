from sklearn.utils import shuffle
sys.path.append('../')
from common import save
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np


class ModelManager:
    def __init__(self, model, preprocess):
        """
        model must have "fit" function, defined like scikit-learn fit, and a attribute for the name
        """
        self.model = model
        self.preprocess = preprocess
    
    def train_model(self, X, y, save_path):
        X, y = self.preprocess_sets(X,y)
        X, y = shuffle(X, y, random_state=42)

        print("\n(num_samples, num_features)")
        print(f"Train dataset : {X.shape}")
        print("\nTraining started")
        self.model.fit(X, y)
        print("Training finished")
        
        self.save_model(save_path)
    
    def test_model(self, X, y):
        X, y = self.preprocess_sets(X,y)
        X, y= shuffle(X, y, random_state=42)

        y_pred = self.model.predict(X)

        print(classification_report(y, y_pred))

        print("Accuray score : " + str(accuracy_score(y, y_pred)))
    
    def preprocess_sets(self, X, y):
        # pre-process latents
        X, y = self.preprocess(X, y)

        y = np.array(y)
        X = np.array(X)
        return X,y

    def save_model(self, save_path):
        print(self.model.name + "will be saved into "+ save_path + " as "+ self.model.name)
        save(self.model, save_path, self.model.name)
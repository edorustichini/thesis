from lightgbm import LGBMClassifier, LGBMModel
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
import sys
sys.path.append('../')
from common import save
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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
        
    
    def learning_curve(self, X, y, save_path):
        from sklearn.model_selection import learning_curve
        import matplotlib.pyplot as plt

        X_train, y_train= self.preprocess_sets(X,y)
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        train_sizes, train_scores, test_scores = learning_curve(self.model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        print("Creating Leaning Curve plot...")
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.title(f'Learning Curve for {self.model.name}')
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(f'{save_path}/learning_curve_{self.model.name}.png')
    
    def test_model(self, X, y):
        X_test, y_test = self.preprocess_sets(X,y)
        
        y_pred = self.model.predict(X_test)

        return y_test, y_pred


    def preprocess_sets(self, X, y):
        # pre-process latents
        X_new, y_new = self.preprocess(X, y)
        del X

        y_new = np.array(y_new)
        X_new = np.array(X_new)
        return X_new,y_new

    def save_model(self, save_path):
        print(self.model.name + "will be saved into "+ save_path + " as "+ self.model.name)
        save(self.model, save_path, self.model.name)
        

class RandomForest(RandomForestClassifier):
    def __init__(self, name=None, **params):
        super().__init__(**params)
        if name is None:
            name = f"RF_{params['n_estimators']}estimators"
        self.name = name
    
class GridSearch(GridSearchCV):
    def __init__(self, estimator, name,params):
        super().__init__(
            estimator=estimator,
            param_grid=params,
            n_jobs=-1,
            cv=3
        )
        if name is None:
            name = f"grid_search"
        self.name = name

class RandomizedSearch(RandomizedSearchCV):
    def __init__(self, estimator, name, params, n_iter=70, random_state=42):
        super().__init__(
            estimator=estimator, 
            param_distributions=params, 
            n_iter=n_iter,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
            cv=3
        )
        if name is None:
            name = f"randomized_search_{n_iter}iter"
        self.name = name

class SVM(SVC):
    def __init__(self, name=None, **params):
        super().__init__(**params)
        if name is None:
            name = f"SVM"
        self.name = name
    

class LGBM(HistGradientBoostingClassifier):
    def __init__(self, name=None, **params):
        # Inizializza il parent normalmente
        super().__init__(**params)

        # Crea il nome del modello
        if name is None:
            name = f"GB"
        self.name = name

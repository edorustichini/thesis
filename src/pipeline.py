import sys
from time import time

from sklearn import clone

from common import format_time
from data.dataset_manager import prepare_dataset
from models.manager import SVM, RandomForest
from models.tester import testing
from models.trainer import training
sys.path.append('../')

import os
from data.preprocessing import flatten_latents, create_patches_dataset, single_patch_per_latent, multiple_patches_per_latent, all_patches_per_latent

class Pipeline:
    def __init__(self):
        self.args = None
        self.coder = None

        self.estimator_type : str = None 
        self.search_type : str = None
        self.model = None

        self.train_preprocess = single_patch_per_latent
        self.test_preprocess = single_patch_per_latent
        
        self.save_latent_path = None
        self.save_late_test_path = None

        self.setup_coder()
    def __init__(self, model = None, train_preprocess=single_patch_per_latent, test_preprocess=single_patch_per_latent):
        self.args = None
        self.coder = None

        self.estimator_type : str = None
        self.search_type : str = None
        self.model = model

        self.train_preprocess = train_preprocess
        self.test_preprocess = test_preprocess

        self.save_latent_path = None
        self.save_latent_test_path = None
    
    def parse_args(self):
        from config import setup_parser
        self.args = setup_parser()

    def setup_coder(self):
        from coder import CoderManager
        coder_manager =  CoderManager(self.args)
        coder_manager.setup()
        self.coder = coder_manager.coder
    
    def determine_model_type(self):
        """Determine which model to use based on args or default to RandomForest"""
        if self.estimator_type is not None:
            return self.estimator_type
        if hasattr(self.args, 'model_name') and self.args.model_name:
            if 'SVM' in self.args.model_name.upper():
                self.estimator_type = 'SVM'
            elif 'RF' in self.args.model_name.upper() or 'RANDOM' in self.args.model_name.upper():
                self.estimator_type = 'RandomForest'
            else:
                self.estimator_type = 'RandomForest'
        else:
            self.estimator_type = 'RandomForest'
        
        print(f"Model type determined: {self.estimator_type}")
        return self.estimator_type
    
    def determine_search_type(self):
        """Determine if we should use grid search, random search, or no search"""
        if self.search_type is not None:
            return self.search_type
        if hasattr(self.args, 'search_type'):
            self.search_type = self.args.search_type
        else:
            # Default behavior - can be overridden
            self.search_type = None
        
        print(f"Search type: {self.search_type if self.search_type else 'No search'}")
        return self.search_type

    def create_model(self, for_search=False):
        """Create model instance based on model_type
        
        Args:
            for_search: If True, returns base estimator for grid/random search
        """
        if self.model is not None:
            return self.model
        if self.estimator_type == 'RandomForest':
            if for_search:
                return RandomForest(random_state=42, oob_score=True, verbose=0)
            else:
                return RandomForest(
                    name=self.args.model_name if hasattr(self.args, 'model_name') else None,
                    random_state=42, 
                    oob_score=True, 
                    verbose=0
                )
        elif self.estimator_type == 'SVM':
            if for_search:
                return SVM(random_state=42, probability=True)
            else:
                return SVM(
                    name=self.args.model_name if hasattr(self.args, 'model_name') else None,
                    random_state=42, 
                    probability=True
                )
        else:
            raise ValueError(f"Unsupported model type: {self.estimator_type}")

    def prepare_dataset(self, df, save_latent_path=None):
        """Prepare training dataset
        
        Returns:
            X_raw: Raw latent representations
            X_hat_raw: Quantized latent representations  
            labels: Labels for the dataset
        """
        if self.save_latent_path:
            print(f"Latents will be saved to: {self.save_latent_path}")
        
        X_raw, X_hat_raw, labels = prepare_dataset(
            self.args,
            self.coder,
            df,
            self.save_latent_path
        )
        print(f"Training data prepared: {len(X_raw)} samples")
        return X_raw, X_hat_raw, labels

    def training(self):
        self.save_latent_path = os.path.join(
            self.args.bin_path,
            str(self.args.set_target_bpp)+"_bpp",
            "latent") if self.args.save_latents else None
        
        X_raw, X_hat_raw, labels = self.prepare_dataset(self.args.train_csv,self.save_latent_path)
        
        print("\nTraining on target y")
        model_y = self.create_model(for_search=(self.search_type is not None))
        start_time = time()
        training(self.args, model_y, X_raw, labels, self.train_preprocess, target='y')
        del X_raw
        end_time = time()
        print("Training time: " + format_time(end_time-start_time))
        print(model_y)

        print("\nTraining on target y_hat")
        model_y_hat = self.create_model(for_search=(self.search_type is not None))
        start_time = time()
        training(self.args, model_y_hat, X_hat_raw, labels, self.test_preprocess, target='y_hat')
        end_time = time()
        print("Training time: " + format_time(end_time-start_time))
        print(model_y_hat)

    def testing(self):
        self.save_latent_test_path = os.path.join(
            self.args.bin_path,
            str(self.args.set_target_bpp)+"_bpp",
            "test",
            "latent") if self.args.save_latents else None

        X_raw, X_hat_raw, labels = self.prepare_dataset(self.args.test_csv, self.save_latent_test_path)

        model = self.create_model(for_search=False)

        print("\nTesting on target y")
        model_y = clone(model)
        testing(model_y, X_raw, labels, self.test_preprocess, target='y')
        del X_raw

        print("\nTesting on target y_hat")
        model_y_hat = clone(model)
        testing(model_y_hat, X_hat_raw, labels, self.test_preprocess, target='y_hat')
        del X_hat_raw

    def run(self):
        self.parse_args()
        self.setup_coder()

        self.training()

        self.testing()

    def run_training_only(self):
        self.parse_args()
        self.setup_coder()

        self.training()
    def run_testing_only(self):
        self.parse_args()
        self.setup_coder()

        self.testing()

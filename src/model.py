import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

@dataclass
class ModelConfig:
    DT_params = {
        "criterion" : ["gini", "entropy"],
        "max_depth" : [None, 2, 5],
        "min_samples_split" : [2, 4, 6]
    }
    random_Forest_params = {
        "n_estimators" : [100, 200],
        "criterion" : ["gini", "entropy"],
        "max_depth" : [None, 3, 5],
        "min_samples_split" : [2, 4, 6],
        "max_features" : [None]
    }
    xgboost_params = {
        "learning_rate" : [1, 0.1, 0.01],
        "max_depth" : [2, 5, 7],
        "n_estimators" : [100, 150, 200]
    }

class DataIngestion:
    def data_ingestion(self, URI):
        """
        Reads the data from the data_path by default,
        custom URI if needed to be provided to the function.
        Returns: X_train, X_test, y_train, y_test
        """
        df = pd.read_csv(URI)
        train_set, test_set = train_test_split(df, test_size=0.30, random_state=45)
        train_set:pd.DataFrame; test_set:pd.DataFrame

        X_train = train_set.drop(columns='Purchased')
        y_train = train_set['Purchased']

        X_test = test_set.drop(columns="Purchased")
        y_test = test_set["Purchased"]

        return X_train, X_test, y_train, y_test

class Preprocessor:
    def initiate_preprocessing(self, X_train, X_test):
        preprocessor = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

        train_data_preprocessed = preprocessor.fit_transform(X_train)
        test_data_preprocessed = preprocessor.transform(X_test)

        # adding preprocessor as attribute
        self.preprocessor_obj = preprocessor

        return train_data_preprocessed, test_data_preprocessed
    

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelConfig()

    def train_model(self, model_name:str, X_train, y_train):
        """
        Trains the model for which the model_name is provided, trains XGBoost by default.
        returns best model, best params
        Returns: {model: best_model, params: best_params)
        """
        model_name = str(model_name).lower()        # enabling case insensitivity
        if model_name=='decisiontree':
            model = DecisionTreeClassifier(random_state=34)
            params = self.model_config.DT_params
        elif model_name=='randomforest':
            model = RandomForestClassifier(random_state=34, verbose=1)
            params = self.model_config.random_Forest_params
        elif model_name=='xgboost':
            model = XGBClassifier(objective='binary:logistic', random_state=34, verbosity=1)
            params = self.model_config.xgboost_params
        else:
            raise ValueError("Incorrect value provided as model name.")
        
        grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy')
        grid.fit(X_train, y_train)
        return {
            "model": grid.best_estimator_,
            "params": grid.best_params_
        }
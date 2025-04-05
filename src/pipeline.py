import mlflow
import mlflow.sklearn
import mlflow.xgboost
import tempfile
from dataclasses import dataclass
from .utils import save_object, evaluate_model
from .model import DataIngestion, Preprocessor, ModelTrainer

@dataclass
class ModelConfig:
    decisiontree = {
        "criterion" : ["gini", "entropy"],
        "max_depth" : [None, 2, 5],
        "min_samples_split" : [2, 4, 6]
    }
    randomforest = {
        "n_estimators" : [100, 200],
        "criterion" : ["gini", "entropy"],
        "max_depth" : [None, 3, 5],
        "min_samples_split" : [2, 4, 6],
        "max_features" : [None]
    }
    xgboost = {
        "learning_rate" : [1, 0.1, 0.01],
        "max_depth" : [2, 5, 7],
        "n_estimators" : [100, 150, 200]
    }


class Pipeline:
    def __init__(self, data_uri, model_name, params):
        self.data_uri = data_uri
        self.model_name = model_name
        self.params =  params
        
    def run_preprocessor_pipeline(self):
        data_ingestion = DataIngestion()
        self.X_train, self.X_test, self.y_train, self.y_test = data_ingestion.data_ingestion(URI=self.data_uri)

        preprocessor = Preprocessor()
        self.X_train, self.X_test = preprocessor.initiate_preprocessing(self.X_train, self.X_test)
        with tempfile.TemporaryDirectory(dir="./artifacts") as temp:
            save_object(f'{temp}/preprocessor.pkl', preprocessor.preprocessor_obj)
            mlflow.log_artifact(f'{temp}/preprocessor.pkl', 'Preprocessor')

    def run_model_pipeline(self):
        model_training = ModelTrainer()
        results = model_training.train_model(self.model_name, self.params, self.X_train, self.y_train)
        
        # making trained_model an instance attribute
        self.trained_model = results.get('model')
        self.input_example = self.X_train[:5]
        
        model_params = results.get('params')
        acc_score, roc_auc = evaluate_model(self.trained_model, self.X_test, self.y_test)
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy score", acc_score)
        mlflow.log_metric("roc auc score", roc_auc)


def run_pipeline(*, data_uri, model_name, params:dict|str):
    model_name = str(model_name).lower()

    if params=="auto":
        cfg = ModelConfig()
        params = getattr(cfg, model_name, {})

    pipeline = Pipeline(data_uri, model_name, params)

    if model_name=="decisiontree":
        mlflow.set_experiment("Decision Tree")
        with mlflow.start_run(description="Training Decision Tree model"):
            pipeline.run_preprocessor_pipeline()
            pipeline.run_model_pipeline()
            mlflow.sklearn.log_model(pipeline.trained_model, "DecisionTree", input_example=pipeline.input_example)
    elif model_name=="randomforest":
        mlflow.set_experiment("Random Forest")
        with mlflow.start_run(description="Training Random Forest model"):
            pipeline.run_preprocessor_pipeline()
            pipeline.run_model_pipeline()
            mlflow.sklearn.log_model(pipeline.trained_model, "RandomForest", input_example=pipeline.input_example)
    elif model_name=="xgboost":
        mlflow.set_experiment("XGBoost")
        with mlflow.start_run(description="Training XGBoost model"):
            pipeline.run_preprocessor_pipeline()
            pipeline.run_model_pipeline()
            mlflow.xgboost.log_model(pipeline.trained_model, "XGBoost", input_example=pipeline.input_example)
    else:
        print("Invalid argument")
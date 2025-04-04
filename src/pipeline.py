import mlflow
import mlflow.sklearn
import mlflow.xgboost
import tempfile
from .utils import save_object, evaluate_model
from .model import DataIngestion, Preprocessor, ModelTrainer

def run_pipeline(*,data_uri, model_name:str):
    data_ingestion = DataIngestion()
    X_train, X_test, y_train, y_test = data_ingestion.data_ingestion(URI=data_uri)

    with mlflow.start_run(description=model_name):
        preprocessor = Preprocessor()
        X_train, X_test = preprocessor.initiate_preprocessing(X_train, X_test)
        with tempfile.TemporaryDirectory(dir="./artifacts") as temp:
            save_object(f'{temp}/preprocessor.pkl', preprocessor.preprocessor_obj)
            mlflow.log_artifact(f'{temp}/preprocessor.pkl', 'Preprocessor')

        model_training = ModelTrainer()
        results = model_training.train_model(model_name, X_train, y_train)
        trained_model = results.get('model')
        model_params = results.get('params')

        acc_score, roc_auc = evaluate_model(trained_model, X_test, y_test)

        input_example = X_train[:5]
        if model_name.lower()=="xgboost":
            mlflow.xgboost.log_model(trained_model, model_name, input_example=input_example)
        else:
            mlflow.sklearn.log_model(trained_model, model_name, input_example=input_example)
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy score", acc_score)
        mlflow.log_metric("roc auc score", roc_auc)
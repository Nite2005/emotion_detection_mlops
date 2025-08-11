import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle
import os
import json
import logging 
from logging.handlers import RotatingFileHandler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import dagshub

# dagshub.init(repo_owner='Nite2005', repo_name='emotion_detection_mlops', mlflow=True)

# dagshub_url = "https://dagshub.com"
# repo_owner = "Nite2005"
# repo_name = "emotion_detection_mlops"

# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = RotatingFileHandler(
    'error.log',
    maxBytes=5*1024,
    backupCount=3
)

file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

def load_test_data(data_path: str) -> pd.DataFrame:
    try:
        test_df = pd.read_csv(data_path)
        logger.debug("Successfully! loaded test data")
        return test_df
    except FileNotFoundError:
        logger.error("File not found in %s", data_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occur : %s", e)
        raise

def load_model(path: str) -> object:
    try:
        model = pickle.load(open(path,"rb"))
        logger.debug("Successfully load model")
        return model
    except FileNotFoundError:
        logger.error("Model not found in %s", path)
        raise
    except Exception as e:
        logger.error("Unexpected error occur in %s", e)


def testing_model(test_df: pd.DataFrame, model: object) -> pd.DataFrame:
    try:
        x_test = test_df.iloc[:,:-1]
        y_test = test_df.iloc[:,-1]
        y_pred = model.predict(x_test)
        return y_pred, y_test, x_test
    except Exception as e:
        logger.error("Unexpected error occur in %s", e)


def metric_evaluation(y_pred: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        print("accuracy_score",accuracy_score(y_test,y_pred))
        print("precission_score",precision_score(y_test,y_pred))
        print("recall_score",recall_score(y_test,y_pred))
        print("f1_score",f1_score(y_test,y_pred))
        metric_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1':f1
        }
        logger.debug("Successfully created metric dictionary ")
        return metric_dict
    except Exception as e:
        logger.error("Unexpected error occur in %s", e)
        raise

def save_metric(path: str, metric_dict: dict) -> None:
    try:
        json.dump(metric_dict, open(path, "w"))
        logger.debug("Succesfully saved metric in reports")
    except Exception as e:
        logger.error("Unexpected error occur in %s", e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run id and path to a json file"""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error occured while saving the model info: %s", e)
        raise

def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            test_data = load_test_data("data/processed/test_df.csv")
            model = load_model("models/model.pkl")
            y_pred, y_test, x_test = testing_model(test_data, model)
            signature = infer_signature(x_test, y_pred)

            metric_dict = metric_evaluation(y_pred, y_test)
            save_metric("reports/metric.json", metric_dict)

            for metric_name, metric_value in metric_dict.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=x_test.head(1)
            )

            save_model_info(run.info.run_id, "model", "reports/experiment_info.json")

            mlflow.log_artifact("reports/metric.json")

            mlflow.log_artifact('reports/experiment_info.json')

            mlflow.log_artifact('error.log')
            logger.debug("Successfully executed model_evaluation process")
            print('Successfully executed model_evaluation process')
        except Exception as e:
            logger.error("Failed to complete model evaluation process")
            print(f"Error{e}")


if __name__ == "__main__":
    main()





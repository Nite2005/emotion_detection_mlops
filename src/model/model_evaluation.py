import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle
import os
import json
import logging 
from logging.handlers import RotatingFileHandler


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
        return y_pred, y_test
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


def main():
    try:
        test_data = load_test_data("data/processed/test_df.csv")
        model = load_model("models/model.pkl")
        y_pred, y_test = testing_model(test_data, model)
        metric_dict = metric_evaluation(y_pred, y_test)
        save_metric("reports/metric.json", metric_dict)
        logger.debug("Successfully executed model_evaluation process")
        print('Successfully executed model_evaluation process')
    except Exception as e:
        logger.error("Failed to complete model evaluation process")
        print(f"Error{e}")


if __name__ == "__main__":
    main()





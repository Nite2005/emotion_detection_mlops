import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import pickle
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger("feature_engineering")
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

def load_data(train_data_path: str, test_data_path:  str) -> pd.DataFrame:
    try:
        train_processed_data = pd.read_csv(train_data_path)
        test_processed_data = pd.read_csv(test_data_path)
        logger.debug("Successfully! loaded processed data")
        return train_processed_data, test_processed_data
    except FileNotFoundError:
        logger.error("File not found in %s or %s", train_data_path, test_data_path )
        raise
    except Exception as e:
        logger.error("Unexpected error occured in %s", e)
        raise


def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    try:
        vectorizer = CountVectorizer(max_features=1000)
        x_train = train_data['content']
        y_train = train_data['sentiment']
        logger.debug("Split train data between x_train and y_train")
        x_test = test_data['content']
        y_test = test_data['sentiment']
        logger.debug("Split test data between x_test and y_test")

        x_train.dropna(inplace=True)
        x_test.dropna(inplace=True)

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        pickle.dump(vectorizer , open("models/vectorizer.pkl",'wb'))
        logger.debug("Successfully! saved vectorizer.pkl file")
        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test
        logger.debug("Successfull! applied CountVectorizer ")
        return train_df, test_df
    except Exception as e:
        logger.error("Unexpected error occured in %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        processed_data_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        train_data.to_csv(
            os.path.join(processed_data_path, "train_df.csv"), index=False
        )
        test_data.to_csv(
            os.path.join(processed_data_path, "test_df.csv"), index=False
        )
        logger.debug("Successfully! saved data into processed folder")
    except Exception as e:
        logger.error("Unexpected error occured in %s", e)
        raise


def main():
    try:
        train_processed_data, test_processed_data = load_data("data/interim/train_processed_data.csv", "data/interim/test_data_processed.csv")
        train_data, test_data = apply_bow(train_processed_data, test_processed_data)
        save_data(train_data, test_data, "./data")
        logger.debug("Successfully! executed feature engineering process")
        print("Successfully executed feature engineering process")
    except Exception as e:
        logger.error("Unexpected error occur : %s", e)
        raise

if __name__ == "__main__":
    main()


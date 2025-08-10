import pandas as pd
import pickle 
from sklearn.linear_model import LogisticRegression
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = RotatingFileHandler(
    'error.log',
    maxBytes=5*1024,
    backupCount=3
)

file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)



def load_train_data(data_path: str) -> pd.DataFrame:
    try:
        train_df = pd.read_csv(data_path)
        logger.debug("Successfully! loaded train data")
        return train_df
    except FileNotFoundError:
        logger.error("File not found in %s", data_path)
        raise

def model_training(train_df: pd.DataFrame) -> object:
    try:
        x_train = train_df.iloc[:,:-1]
        y_train = train_df.iloc[:,-1]
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        logger.debug("Model is trained successfully")
        return clf
    except Exception as e:
        logger.error("Unexpected error occur: %s", e)
        raise


def save_model(model_name: object, path: str) -> None:
    try:
        pickle.dump(model_name, open(path,"wb"))
        logger.debug("Successfully saved model in %s", path)
    except Exception as e:
        logger.error("Unexpected error occur in %s", e)
        raise
def main():
    try:
        train_df = load_train_data("data/processed/train_df.csv")
        model = model_training(train_df)
        save_model(model, "models/model.pkl")
        print("Successfully executed model building process")
    except Exception as e:
        logger.error("Failed to complete model building process")
        print(f"Error:{e}")

if __name__ == "__main__":
    main()

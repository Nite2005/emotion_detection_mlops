import numpy as np 
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


file_handler = RotatingFileHandler(
    "error.log",
    maxBytes=5*1024,
    backupCount=3
)


file_handler.setLevel('ERROR')


formatter = logging.Formatter(
    '%(asctime)s - %(name)s -%(levelname)s - %(message)s'
)


console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


nltk.download('wordnet')
nltk.download('stopwords')


def load_data(params_path: str) -> pd.DataFrame:
    """Load data from data/raw folder"""
    try:
        train_data = pd.read_csv(f"{params_path}/train.csv")
        test_data = pd.read_csv(f"{params_path}/test.csv")
        logger.debug("Successfully! Data loaded from data/raw folder ")
        return train_data, test_data
    except FileNotFoundError:
        logger.error("File not found in %s", params_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occured %s", e)
        raise


def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        logger.debug("text is converted in lower case")
        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug("Stop words removed from text")
        df['content'] = df['content'].apply(removing_numbers)
        logger.debug("All the numbers are removed from text")
        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug("All the punctuations are removed from the text")
        df['content'] = df['content'].apply(removing_urls)
        logger.debug("All the urls are removed from the text")
        df['content'] = df['content'].apply(lemmatization)
        logger.debug("Lematization has applied on the text")
        logger.debug("Successfully! Data have normalize")
        return df
    except Exception as e:
        logger.error("Unexpected error occured %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save processed data into the folder of data/interim"""
    try:
        interim_data_path = os.path.join(data_path,'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        train_data.to_csv(
            os.path.join(interim_data_path, "train_processed_data.csv"), index=False
        )
        test_data.to_csv(
            os.path.join(interim_data_path, "test_data_processed.csv"), index=False
        )
        logger.debug("Train_processed data and test processed data saved in %s", interim_data_path)
    except Exception as e:
        logger.error("Unexpected error occured in %s", e)
        raise

    
def main():
    try:    
        train_data, test_data = load_data("data/raw")
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        save_data(train_processed_data, test_processed_data, data_path="./data")
        logger.debug("Successfully completed data_preprocessing process")
    except Exception as e:
        logging.error("Unexpected error occured in %s", e)
        print("Failed to process data_preprocessing process: %s", e)


if __name__ == '__main__':
    main()
    

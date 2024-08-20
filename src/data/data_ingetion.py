import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import yaml

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_params(params_path: str) -> float:
    logger.info('Loading parameters from %s', params_path)
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.info('Parameters loaded successfully')
            return test_size
    except FileNotFoundError:
        logger.error(f"Error: the file {params_path} was not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error: failed to parse the YAML file {params_path}")
        raise
    except Exception as e:
        logger.error(f"Error: an unexpected error occurred while loading {params_path}")
        raise

def load_data(path: str) -> pd.DataFrame:
    logger.info("Starting to load data from %s", path)
    try:
        df = pd.read_csv(path)
        logger.debug(f"Data loaded successfully from {path}")
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing the dataset")
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    logger.info("Starting to save the data")
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        test_size = load_params(params_path='params.yaml')
        df = load_data(path='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

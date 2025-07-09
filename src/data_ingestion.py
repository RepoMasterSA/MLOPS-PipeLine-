import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# -------------------------- Logging Setup --------------------------

# Create a log directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File logging
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Log format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------------- Utility Functions --------------------------

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML config file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters successfully loaded from %s', params_path)
            return params
    except FileNotFoundError:
        logger.error('YAML file not found at path: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Failed to parse YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading parameters: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file or URL."""
    try:
        # Fix: direct GitHub links need ?raw=true to serve raw CSV
        if "github.com" in data_url and "raw" not in data_url:
            data_url = data_url.replace("blob/", "").replace("github.com", "raw.githubusercontent.com")

        df = pd.read_csv(data_url)
        logger.debug('Data successfully loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('CSV parsing error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and structure the raw dataset."""
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed successfully')
        return df
    except KeyError as e:
        logger.error('Missing expected column in dataset: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during data preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save processed train and test data to disk."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Train and test datasets saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving datasets: %s', e)
        raise

# -------------------------- Main Ingestion Process --------------------------

def main():
    try:
        logger.info("Data ingestion process started.")
        
        # Load parameters from YAML file
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        # Load and preprocess data
        data_path = 'https://github.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/blob/main/experiments/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)

        # Split data into training and test sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the split datasets
        save_data(train_data, test_data, data_path='./data')

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.error('Data ingestion failed: %s', e)
        print(f"❌ Error: {e}")

# -------------------------- Entry Point --------------------------

if __name__ == '__main__':
    main()

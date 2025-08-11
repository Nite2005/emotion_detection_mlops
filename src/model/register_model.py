import json
import mlflow
import logging
import os
import dagshub

dagshub.init(repo_owner='Nite2005', repo_name='emotion_detection_mlops', mlflow=True)

dagshub_url = "https://dagshub.com"
repo_owner = "Nite2005"
repo_name = "emotion_detection_mlops"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
logger = logging.getLogger('register_model')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a json file"""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from  %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found in %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the mlflow Model Registry """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version,
            stage = 'Staging'
        )

        logger.debug(f"Model {model_name} version {model_version.version} registered and transitioned to staging")
    except Exception as e:
        logger.error("Error during model registration: %s", e)


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
        logger.debug("Successfully registed model in mlflow")
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
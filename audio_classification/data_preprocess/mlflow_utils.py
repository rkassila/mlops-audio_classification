import os
import time
import torch
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import yaml

# Path to your config file
CONFIG_PATH = "config/mlflow_config.yaml"

# Function to load the YAML configuration
def load_mlflow_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config['mlflow']

# Load the MLflow configuration
mlflow_config = load_mlflow_config()

MLFLOW_TRACKING_URI = mlflow_config['tracking_uri']
MLFLOW_EXPERIMENT = mlflow_config['experiment_name']
MLFLOW_MODEL_NAME = mlflow_config['model_name']
LOCAL_REGISTRY_PATH = mlflow_config['local_registry_path']  # Added local registry path

# Set MLflow tracking URI globally
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics to MLflow and locally.
    """
    # Log to MLflow
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metrics(metrics)
    print("✅ Results saved on MLflow")


def save_model(model: torch.nn.Module) -> None:
    """
    Save the model only to MLflow.
    """
    # Log the model directly to MLflow (no local saving)
    mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=MLFLOW_MODEL_NAME)
    print("✅ Model saved to MLflow")


def load_model(version=None) -> torch.nn.Module:
    """
    Load a saved model from MLflow by version (latest if not provided).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        if version is None:
            # Get the latest version of the model
            latest_version_info = client.get_latest_versions(name=MLFLOW_MODEL_NAME)[0]
            version = latest_version_info.version

        # Load the model from MLflow
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{version}"
        model = mlflow.pytorch.load_model(model_uri)
        print(f"✅ Model version {version} loaded from MLflow")
        return model
    except Exception as e:
        print(f"❌ No model found: {e}")
        return None


def mlflow_transition_model_if_better(new_metrics: dict):
    """
    Transition the model to 'Staging' only if its performance is better than the model in 'Staging' or 'Production'.
    If no model exists in 'Staging' or 'Production', the new model is transitioned to 'Staging'.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Fetch metrics of the current models in Staging and Production
    current_metrics_staging = get_model_metrics(stage="Staging")
    current_metrics_production = get_model_metrics(stage="Production")

    new_accuracy = new_metrics.get("Test Accuracy", 0)
    new_mae = new_metrics.get("Test MAE", float('inf'))

    # If there is no model in Staging or Production, promote the new model to Staging
    if current_metrics_staging is None and current_metrics_production is None:
        print(f"❌ No model in Staging or Production. Transitioning new model to Staging...")
        transition_model(client, current_stage="None", new_stage="Staging")
        return

    # Check if the new model is better than the one in Staging or Production
    staging_accuracy = current_metrics_staging.get("Test Accuracy", 0) if current_metrics_staging else 0
    staging_mae = current_metrics_staging.get("Test MAE", float('inf')) if current_metrics_staging else float('inf')

    production_accuracy = current_metrics_production.get("Test Accuracy", 0) if current_metrics_production else 0
    production_mae = current_metrics_production.get("Test MAE", float('inf')) if current_metrics_production else float('inf')

    # Compare new model with the better of Staging or Production
    if new_accuracy > max(staging_accuracy, production_accuracy) or new_mae < min(staging_mae, production_mae):
        print(f"✅ New model is better. Transitioning to Staging...")
        transition_model(client, current_stage="None", new_stage="Staging")
    else:
        print(f"❌ New model is not better than the current model in Staging or Production. No transition will be made.")


def get_model_metrics(stage=None, version=None):
    """
    Fetch the test metrics (like accuracy or MAE) for the latest or specified model version and stage.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        if version is None:
            # Get the latest version of the model from the specified stage
            if stage:
                versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
                if not versions:
                    return None
                version = versions[0].version

        # Get the run ID associated with the version
        run_id = client.get_model_version(MLFLOW_MODEL_NAME, version).run_id
        metrics = mlflow.get_run(run_id).data.metrics
        return metrics
    except Exception as e:
        print(f"❌ No model metrics found: {e}")
        return None


def transition_model(client: MlflowClient, current_stage: str, new_stage: str):
    """
    Transition the latest model from the `current_stage` to the `new_stage`.
    """
    try:
        version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])
        if not version:
            print(f"❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
            return None

        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=version[0].version,
            stage=new_stage,
            archive_existing_versions=True
        )
        print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")
    except Exception as e:
        print(f"❌ Error while transitioning model: {e}")


import mlflow
from mlflow.tracking import MlflowClient

def mlflow_run(func):
    """
    Decorator for running functions with MLflow auto-logging.
    Ensures the experiment exists and is active.
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()  # End any existing run to avoid conflicts
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = MlflowClient()

        # Try to retrieve the experiment by name
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)

        # Check if experiment exists or is deleted
        if experiment is None:
            print(f"Experiment '{MLFLOW_EXPERIMENT}' does not exist. Creating a new one.")
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT)
        elif experiment.lifecycle_stage == 'deleted':
            print(f"Experiment '{MLFLOW_EXPERIMENT}' is deleted. Re-creating it.")
            client.delete_experiment(experiment.experiment_id)
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT)
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment '{MLFLOW_EXPERIMENT}' with ID: {experiment_id}")

        # Set the experiment for the current run
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        # Start the MLflow run with the experiment ID
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.pytorch.autolog()  # Enable auto-logging for PyTorch
            result = func(*args, **kwargs)  # Execute the function

        print("✅ MLflow run completed")
        return result

    return wrapper

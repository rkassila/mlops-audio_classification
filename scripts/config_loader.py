import yaml


def load_datasets_config(config_path="config/datasets_config.yaml"):
    """
    Load dataset configurations from a YAML file.

    Args:
        config_path (str): Path to the datasets config file.
        Defaults to "config/datasets_config.yaml".

    Returns:
        list: A list of dictionaries containing dataset configuration details.
    """
    with open(config_path, 'r') as config_file:
        datasets_config = yaml.safe_load(config_file)['datasets']
    return datasets_config


def load_gcs_config(config_path="config/gcs_config.yaml"):
    """
    Load Google Cloud Storage (GCS) configurations from a YAML file.

    Args:
        config_path (str): Path to the GCS config file.
        Defaults to "config/gcs_config.yaml".

    Returns:
        dict: A dictionary containing GCS config details like 'bucket_name'.
    """
    with open(config_path, 'r') as config_file:
        gcs_config = yaml.safe_load(config_file)
    return gcs_config

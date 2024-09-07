import requests
import pyarrow.parquet as pq
from datasets import load_dataset
from io import BytesIO


def load_dataset_from_subset(dataset_config):
    """
    Load a specific dataset subset using the Hugging Face `datasets` library.

    Args:
        dataset_config (dict): Configuration details for the dataset,
        including 'name' and 'subset'.

    Returns:
        tuple: A tuple containing:
            - dataset_split (Dataset): Split of the dataset (usually 'train').
            - parquet_filename (str): Name of the Parquet file to be generated.
    """
    dataset = load_dataset(dataset_config["name"], name=dataset_config["subset"], split=dataset_config['split'])
    print(f"Loaded {dataset_config['name']}/{dataset_config['subset']}.")

    parquet_filename = f"{dataset_config['subset']}.parquet"
    return dataset, parquet_filename


def fetch_parquet_file_urls(dataset_config):
    """
    Generate URLs for downloading specific Parquet files from Hugging Face.

    Args:
        dataset_config (dict): Configuration details,
        including the dataset 'name' and 'files'.

    Returns:
        list: A list of fully constructed file URLs.

    Raises:
        ValueError: If no file names are found in the configuration.
    """
    base_url = (
        f"https://huggingface.co/datasets/{dataset_config['name']}/resolve/"
        "main/data"
    )

    file_names = dataset_config.get('files', [])

    if not file_names:
        raise ValueError("No file names found for this dataset.")

    file_urls = [f"{base_url}/{file_name}" for file_name in file_names]
    return file_urls


def get_parquet_file_metadata(file_url):
    """
    Fetch and extract metadata from a remote Parquet file.

    Args:
        file_url (str): The URL of the Parquet file to be fetched.

    Returns:
        tuple: A tuple containing:
            - num_rows (int): The total number of rows in the Parquet file.
            - features (list of str): List of column names in the Parquet file.

    Raises:
        requests.exceptions.RequestException:
        If the request to fetch the Parquet file fails.
        ValueError: If there is an issue reading the Parquet file structure.
    """
    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    file_data = BytesIO(response.content)
    parquet_file = pq.ParquetFile(file_data)

    num_rows = parquet_file.metadata.num_rows
    features = parquet_file.schema.names

    return num_rows, features

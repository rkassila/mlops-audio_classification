import requests
from google.cloud import storage
from io import BytesIO


def stream_and_upload_parquet(file_url, bucket_name, dataset_name, file_name):
    """
    Streams a Parquet file from a given URL and
    uploads it directly to Google Cloud Storage (GCS).

    Args:
        file_url (str): URL of the Parquet file to be streamed and uploaded.
        bucket_name (str): Name of GCS bucket where the file will be stored.
        dataset_name (str): The name of the dataset, used as a folder in GCS.
        file_name (str): The name of the file to be stored in GCS.

    Returns:
        str: The GCS path where the file is uploaded.

    Raises:
        requests.exceptions.RequestException: If issue during the HTTP request.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{dataset_name}/{file_name}")

    print(f"Streaming and uploading {file_name} to GCS...")

    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    blob.upload_from_file(
        BytesIO(response.content),
        content_type="application/octet-stream"
    )
    gcs_file_path = f"gs://{bucket_name}/{dataset_name}/{file_name}"

    print(f"Uploaded {file_name} to {gcs_file_path}")
    return gcs_file_path


def upload_parquet_to_gcs(
    dataset_split, bucket_name,
    dataset_name, parquet_filename
):
    """
    Converts a dataset split into Parquet format and
    uploads it to Google Cloud Storage (GCS).

    Args:
        dataset_split (datasets.Dataset): Dataset to be converted and uploaded.
        bucket_name (str): GCS bucket where the Parquet file will be stored.
        dataset_name (str): The name of the dataset, used as a folder in GCS.
        parquet_filename (str): The name of the file to be stored in GCS.

    Returns:
        str: The GCS path where the Parquet file is uploaded.

    Raises:
        Exception: If there is an error during conversion or file upload.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{dataset_name}/{parquet_filename}")

    print(f"Converting to Parquet, uploading {parquet_filename} to GCS...")

    parquet_buffer = BytesIO()
    dataset_split.to_parquet(parquet_buffer)
    parquet_buffer.seek(0)

    print(f"Uploading {parquet_filename} to GCS...")
    blob.upload_from_file(
        parquet_buffer,
        content_type="application/octet-stream"
    )
    gcs_file_path = f"gs://{bucket_name}/{dataset_name}/{parquet_filename}"

    print(f"Uploaded Parquet file to {gcs_file_path}")
    return gcs_file_path

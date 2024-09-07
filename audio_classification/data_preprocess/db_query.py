import psycopg2
from google.cloud import storage
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# PostgreSQL connection settings
pg_conn_params = {
    'dbname': 'postgres',
    'user': 'dataengineer',
    'password': 'postgre-challenge-waTpof',
    'host': '34.78.172.238',
    'port': '5432'
}


def query_postgres(dataset_name):
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**pg_conn_params)
        cursor = conn.cursor()

        # Query to get the GCS path for a dataset
        cursor.execute("""
            SELECT f.gcs_link
            FROM files f
            JOIN datasets d ON f.dataset_id = d.id
            WHERE d.name = %s;
        """, (dataset_name,))

        # Fetch the GCS links
        gcs_links = cursor.fetchall()

        # Close the PostgreSQL connection
        cursor.close()
        conn.close()

        return gcs_links

    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return []

def load_data_from_gcs(gcs_links):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Load all files associated with the dataset into a single pandas DataFrame
    dataframes = []
    for gcs_link in gcs_links:
        gcs_file_path = gcs_link[0].replace('gs://', '')
        bucket_name, *blob_path = gcs_file_path.split('/', 1)
        blob_path = blob_path[0]

        # Get the bucket and blob (file)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download the file to a local file-like object
        file_obj = blob.download_as_bytes()

        # Read the Parquet file into a DataFrame
        table = pq.read_table(pa.BufferReader(file_obj))
        df = table.to_pandas()
        dataframes.append(df)

    # Concatenate all DataFrames into one
    full_dataset = pd.concat(dataframes, ignore_index=True)
    return full_dataset

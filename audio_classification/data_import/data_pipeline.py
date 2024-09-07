from audio_classification.data_import.gcs_utils import stream_and_upload_parquet, upload_parquet_to_gcs
from audio_classification.data_import.config_loader import load_datasets_config, load_gcs_config
from audio_classification.data_import.datasets_utils import (
    load_dataset_from_subset,
    fetch_parquet_file_urls,
    get_parquet_file_metadata
)
from audio_classification.data_import.postgresql_utils import (
    get_postgresql_connection,
    initialize_metadata_tables,
    check_existing_dataset,
    delete_existing_dataset,
    update_dataset_metadata,
    insert_dataset_entry,
    get_dataset_id,
    insert_file_metadata
)


def ask_user_for_rewrite(conn, table_name):
    """
    Prompt the user to decide to overwrite an existing dataset in PostgreSQL.
    If yes, the existing dataset and its associated files are deleted.

    Args:
        conn (psycopg2.connection): Connection obj with the PostgreSQL db.
        table_name (str): The name of the dataset to potentially overwrite.

    Returns:
        bool: True if the user wants to overwrite the existing dataset.
    """
    while True:
        user_input = input(
            f"The dataset '{table_name}' already exists in PostgreSQL.\n"
            "Do you want to rewrite it? (yes/no): "
        ).strip().lower()

        if user_input in ['yes', 'no']:
            if user_input == 'yes':
                delete_existing_dataset(conn, table_name)
            return user_input == 'yes'
        else:
            print("Please enter 'yes' or 'no'.")


def catalog_dataset_metadata(dataset_config, conn, gcs_config):
    """
    Stores dataset metadata in PostgreSQL and uploads its files to GCS.

    Args:
        dataset_config (dict): Configuration details for the dataset.
        conn (psycopg2.connection): Obj for interacting with the PostgreSQL db.
        gcs_config (dict): Configuration details for Google Cloud Storage.

    Returns:
        None
    """
    bucket_name = gcs_config['bucket_name']
    dataset_name = dataset_config['name']
    table_name = dataset_config['table_name']
    description = dataset_config.get('description', 'No description provided')

    # Check if the dataset already exists in PostgreSQL
    if check_existing_dataset(conn, table_name):
        if not ask_user_for_rewrite(conn, table_name):
            print(f"Skipping dataset '{dataset_name}' as requested by user.")
            return

    print(f"Cataloging dataset '{dataset_name}'...")

    # Ensure dataset entry is created in datasets table before processing files
    insert_dataset_entry(conn, table_name, description)

    if dataset_config.get('files_to_download'):
        # Handle file-based datasets
        file_urls = fetch_parquet_file_urls(dataset_config)
        total_num_rows = 0
        features = []

        # Get the dataset_id from the datasets table for referencing in files
        dataset_id = get_dataset_id(conn, table_name)

        for index, file_url in enumerate(file_urls):
            file_name = file_url.split("/")[-1]
            gcs_file_path = stream_and_upload_parquet(
                file_url, bucket_name, dataset_name, file_name)

            # Get the metadata (number of rows and features) for Parquet files
            num_rows, file_features = get_parquet_file_metadata(file_url)

            if index == 0:
                # Store features once (assuming consistent structure for files)
                features = file_features

            total_num_rows += num_rows

            # Store metadata in the PostgreSQL catalog for each file
            insert_file_metadata(
                conn,
                dataset_id,
                file_name,
                gcs_file_path,
                num_rows,
                file_features
            )
            print(f"Writing metadata for file '{file_name}' to PostgreSQL...")

        # Update the overall dataset metadata in the datasets table
        update_dataset_metadata(
            conn,
            table_name,
            total_num_rows,
            features,
            description
        )

    else:
        # Handle subset-based datasets using the datasets library
        dataset_split, parquet_filename = load_dataset_from_subset(
            dataset_config
        )

        if dataset_split:
            gcs_file_path = upload_parquet_to_gcs(
                dataset_split,
                bucket_name,
                dataset_name,
                parquet_filename
            )

            # Calculate number of rows and extract features for the subset
            num_rows = dataset_split.num_rows
            features = dataset_split.column_names

            # Store the metadata in the datasets table
            update_dataset_metadata(
                conn,
                table_name,
                num_rows,
                features,
                description
            )

            # Also store the file-level metadata
            insert_file_metadata(
                conn,
                get_dataset_id(conn, table_name),
                parquet_filename,
                gcs_file_path, num_rows,
                features
            )
            print(
                f"Cataloged {dataset_name}/{dataset_config['subset']}."
            )


def run_data_pipeline():
    """
    Execute the data cataloging pipeline.

    This function orchestrates the entire process of loading configurations,
    establishing db connection, and cataloging multiple datasets in PostgreSQL.
    It handles subset-based and file-based datasets, storing their metadata
    and uploading them to Google Cloud Storage (GCS) as needed.

    Steps:
        1. Load dataset, GCS, and PostgreSQL configurations from YAML files.
        2. Establish a connection to the PostgreSQL database.
        3. Initialize the metadata tables in the database if they do not exist.
        4. Iterate through the datasets in the configuration:
            - Skip any invalid configurations.
            - Attempt to catalog each dataset and handle any errors that arise.
        5. Close the PostgreSQL connection once processing is complete.

    Returns:
        None
    """
    datasets_config = load_datasets_config()
    gcs_config = load_gcs_config()['gcs']
    postgresql_config = load_gcs_config()['postgresql']

    conn = get_postgresql_connection(postgresql_config)
    initialize_metadata_tables(conn)

    for dataset_config in datasets_config:
        if dataset_config is None or not isinstance(dataset_config, dict):
            print(f"Skipping invalid dataset configuration: {dataset_config}")
            continue

        try:
            catalog_dataset_metadata(dataset_config, conn, gcs_config)
        except Exception as e:
            print(
                f"Error processing {dataset_config.get('name', 'unknown')} {e}"
            )

    conn.close()

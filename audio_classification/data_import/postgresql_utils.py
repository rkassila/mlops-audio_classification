import psycopg2


def get_postgresql_connection(config):
    """
    Establish a connection to the PostgreSQL db using provided configuration.

    Args:
        config (dict): A dictionary containing PostgreSQL connection details
        such as host, port, dbname, user, and password.

    Returns:
        psycopg2.connection: A connection object for the PostgreSQL database.

    Raises:
        Exception: If the connection to the database fails.
    """
    print(
        f"Connecting to PostgreSQL database '{config['dbname']}' at "
        f"{config['host']}:{config['port']} as user '{config['user']}'..."
    )
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            dbname=config['dbname'],
            user=config['user'],
            password=config['password']
        )
        print("Connection to PostgreSQL established successfully!")
        return conn
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        raise


def initialize_metadata_tables(conn):
    """
    Initialize and create the required metadata tables in the PostgreSQL db.
    This function creates the 'datasets' and 'files' tables and
    ensures all necessary columns exist.

    Args:
        conn (psycopg2.connection): A connection object for the PostgreSQL db.
    """
    cursor = conn.cursor()

    # Create the datasets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE,
        description TEXT,
        num_rows BIGINT,
        features TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create the files table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id SERIAL PRIMARY KEY,
        dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
        file_path TEXT,
        gcs_link TEXT,
        num_rows BIGINT,
        features TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Ensure that all required columns exist (for schema upgrades)
    columns_to_ensure = [
        ("datasets", "num_rows", "BIGINT"),
        ("datasets", "features", "TEXT"),
        ("files", "num_rows", "BIGINT"),
        ("files", "features", "TEXT"),
        ("files", "gcs_link", "TEXT")
    ]

    for table, column, column_type in columns_to_ensure:
        cursor.execute(f"""
        DO $$ BEGIN
            BEGIN
                ALTER TABLE {table} ADD COLUMN {column} {column_type};
            EXCEPTION
                WHEN duplicate_column THEN
                    RAISE NOTICE 'Column {column} already exists in {table}.';

            END;
        END $$;
        """)

    conn.commit()
    cursor.close()


def check_existing_dataset(conn, table_name):
    """
    Check if a dataset already exists in the PostgreSQL database.

    Args:
        conn (psycopg2.connection): A connection object for the PostgreSQL db.
        table_name (str): The name of the dataset to check.

    Returns:
        bool: True if the dataset exists, False otherwise.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM datasets WHERE name = %s)",
        (table_name,)
    )
    exists = cursor.fetchone()[0]
    cursor.close()
    return exists


def delete_existing_dataset(conn, table_name):
    """
    Delete a dataset and its associated files from the PostgreSQL database.

    Args:
        conn (psycopg2.connection): A connection object for the PostgreSQL db.
        table_name (str): The name of the dataset to delete.
    """
    cursor = conn.cursor()

    # Delete all file records related to the dataset
    cursor.execute("""
        DELETE FROM files
        WHERE dataset_id = (SELECT id FROM datasets WHERE name = %s)
    """, (table_name,))

    # Delete the dataset itself
    cursor.execute("""
        DELETE FROM datasets
        WHERE name = %s
    """, (table_name,))

    conn.commit()
    cursor.close()

    print(f"Deleted the existing dataset '{table_name}' from PostgreSQL.")


def update_dataset_metadata(conn, table_name, num_rows, features, description):
    """
    Update the metadata of a dataset in the PostgreSQL 'datasets' table.

    Args:
        conn (psycopg2.connection): Connection object for the PostgreSQL db.
        table_name (str): The name of the dataset to update.
        num_rows (int): The updated number of rows in the dataset.
        features (list): The updated list of feature/column names.
        description (str): The updated description of the dataset.
    """
    features_str = ', '.join(features)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE datasets
        SET num_rows = %s, features = %s, description = %s
        WHERE name = %s
    """, (num_rows, features_str, description, table_name))
    conn.commit()
    cursor.close()


def insert_dataset_entry(conn, table_name, description):
    """
    Insert a dataset entry into the PostgreSQL 'datasets' table
    if it doesn't already exist.

    Args:
        conn (psycopg2.connection): Connection object for the PostgreSQL db.
        table_name (str): The name of the dataset to insert.
        description (str): The description of the dataset.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO datasets (name, description, num_rows, features)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name) DO NOTHING
    """, (table_name, description, 0, ""))
    conn.commit()
    cursor.close()


def get_dataset_id(conn, table_name):
    """
    Retrieve the ID of a dataset from the PostgreSQL 'datasets' table.

    Args:
        conn (psycopg2.connection): Connection object for the PostgreSQL db.
        table_name (str): The name of the dataset to fetch the ID for.

    Returns:
        int: The ID of the dataset.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM datasets WHERE name = %s", (table_name,))
    dataset_id = cursor.fetchone()[0]
    cursor.close()
    return dataset_id


def insert_file_metadata(
    conn, dataset_id, file_name,
    gcs_file_path, num_rows, file_features
):
    """
    Insert file metadata into the PostgreSQL 'files' table.

    Args:
        conn (psycopg2.connection): Connection object for the PostgreSQL db.
        dataset_id (int): The ID of the dataset the file belongs to.
        file_name (str): The name of the file.
        gcs_file_path (str): The Google Cloud Storage path of the file.
        num_rows (int): The number of rows in the file.
        file_features (list): The list of features/column names in the file.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO files (dataset_id, file_path, gcs_link, num_rows, features)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        dataset_id, file_name,
        gcs_file_path, num_rows, ', '.join(file_features)
        ))
    conn.commit()
    cursor.close()

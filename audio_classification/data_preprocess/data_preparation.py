import pandas as pd
from db_query import query_postgres, load_data_from_gcs
from audio_dataset import split_dataset
import torch

# Function to load the dataset from PostgreSQL and Google Cloud Storage
def load_and_prepare_data(dataset_name):
    print(f"Querying PostgreSQL for dataset: {dataset_name}...")
    # Get the dataset GCS links from PostgreSQL
    gcs_links = query_postgres(dataset_name)

    print(f"Loading data from Google Cloud Storage...")
    # Load the data from GCS
    full_dataset = load_data_from_gcs(gcs_links)

    print("Preparing the dataset...")
    # Prepare the DataFrame (dropping unnecessary columns)
    pd_df = pd.DataFrame(full_dataset)
    pd_df.drop(columns=['audio', 'english_transcription', 'lang_id'], inplace=True)

    print("Data loaded and prepared.")
    return pd_df

# Function to split data into train and test features and labels
def create_train_test_split(pd_df):
    print("Splitting dataset into training and testing sets...")
    # Split the dataset into DataLoaders
    train_dataloader, test_dataloader = split_dataset(pd_df)

    X_train, y_train, X_test, y_test = [], [], [], []

    # Iterate through train DataLoader to get X_train and y_train
    print("Processing training data...")
    for features, labels in train_dataloader:
        X_train.append(features)
        y_train.append(labels)

    # Iterate through test DataLoader to get X_test and y_test
    print("Processing testing data...")
    for features, labels in test_dataloader:
        X_test.append(features)
        y_test.append(labels)

    # Convert lists to tensors
    print("Converting data to tensors...")
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)

    print(f"Data processed. Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

# Main function to load, prepare, and split the dataset
def main():
    # Specify the dataset name
    dataset_name = 'minds14'  # Replace with your dataset name

    # Load and prepare the data
    pd_df = load_and_prepare_data(dataset_name)

    # Create X and y for training and testing
    X_train, X_test, y_train, y_test = create_train_test_split(pd_df)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print("Starting data preparation...")
    X_train, X_test, y_train, y_test = main()
    print("Data preparation complete.")

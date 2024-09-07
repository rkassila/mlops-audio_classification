import pandas as pd
from db_query import query_postgres, load_data_from_gcs
from audio_dataset import split_dataset

# Get the dataset GCS links from PostgreSQL
dataset_name = 'minds14'  # Replace with desired dataset
gcs_links = query_postgres(dataset_name)

# Load the data from GCS
full_dataset = load_data_from_gcs(gcs_links)

# Prepare the DataFrame
pd_df = pd.DataFrame(full_dataset)
pd_df.drop(columns=['audio', 'english_transcription', 'lang_id'], inplace=True)

# Split the dataset and create DataLoaders
train_dataloader, test_dataloader = split_dataset(pd_df)

# The data is now preprocessed and ready for further use

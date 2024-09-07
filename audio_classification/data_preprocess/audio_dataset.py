import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from audio_features import preprocess_audio

class AudioDataset(Dataset):
    def __init__(self, df, target_length=100, n_mfcc=13):
        self.paths = df['path'].values
        self.labels = df['intent_class'].values
        self.target_length = target_length
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        mfcc_features = preprocess_audio(audio_path, target_length=self.target_length, n_mfcc=self.n_mfcc)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc_features, label

def split_dataset(pd_df):
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(pd_df, test_size=0.2, random_state=42)

    # Create the training and testing datasets
    train_dataset = AudioDataset(train_df)
    test_dataset = AudioDataset(test_df)

    # Create DataLoaders for train and test
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataloader, test_dataloader

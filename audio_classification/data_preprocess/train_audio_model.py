import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error
from data_preparation import main
from mlflow_utils import save_model, save_results, mlflow_transition_model_if_better


class AudioCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(64 * (input_height // 8) * (input_width // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), 1, 40, -1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    return total_loss / len(test_loader), accuracy, mae


def main_training():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = main()

    input_height = 40
    input_width = X_train.shape[1] // input_height
    num_classes = len(torch.unique(y_train))
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AudioCNN(input_height=input_height, input_width=input_width, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, criterion, optimizer, train_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # After all epochs, evaluate the model and log results in MLflow
    test_loss, test_acc, test_mae = evaluate_model(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test MAE: {test_mae:.4f}")

    # Log metrics to MLflow
    metrics = {"Train Loss": train_loss, "Train Accuracy": train_acc,
               "Test Loss": test_loss, "Test Accuracy": test_acc, "Test MAE": test_mae}
    save_results(params=None, metrics=metrics)

    # Save the model
    save_model(model)

    # After training, transition the model to `Staging` only if it's better
    mlflow_transition_model_if_better(new_metrics=metrics)


if __name__ == '__main__':
    main_training()

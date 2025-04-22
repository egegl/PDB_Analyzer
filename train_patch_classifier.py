import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_dir):
    """Load all per-patch CSVs from data_dir/*/*_patches.csv."""
    pattern = os.path.join(data_dir, "*", "*_patches.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No patch CSV files found with pattern: {pattern}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(description="Train patch classifier")
    parser.add_argument("--data-dir", default="out", help="Directory containing patch CSV subdirs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Load dataset
    df = load_data(args.data_dir)
    feature_cols = ["CX", "ASA", "Hydrophobicity", "Planarity", "Roughness"]
    required_cols = feature_cols + ["Type"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSVs")

    X = df[feature_cols].values
    # Map labels: Surface -> 0, Interface -> 1
    df["Label"] = df["Type"].map({"Surface": 0, "Interface": 1})
    if df["Label"].isnull().any():
        raise ValueError("Some patch types are not 'Surface' or 'Interface'")
    y = df["Label"].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Data loaders
    train_ds = PatchDataset(X_train, y_train)
    test_ds = PatchDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    model = MLP(input_dim=len(feature_cols)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")

    # Evaluation on test set
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    test_acc = total_correct / total_samples
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "patch_classifier.pt")
    print("Saved trained model to patch_classifier.pt")

if __name__ == "__main__":
    main()

# train_particle_classifier_two_stage.py

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Config
CSV_PATH = "./merged_output.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS_STAGE1 = 10
NUM_EPOCHS_STAGE2 = 5
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CLASSIFIER_PATH = "particle_classifier.pt"
MODEL_REGRESSOR_PATH = "particle_regressor.pt"

# Dataset
class MultiModalDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.scaler = StandardScaler()
        self.numeric_data = self.scaler.fit_transform(
            self.data[["scatter_value", "concentration1", "concentration2"]].values
        )
        self.labels_class = self.data["num_particles"].values.astype(int) - 1
        self.labels_reg = self.data[["concentration1", "concentration2"]].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        numeric = torch.tensor(self.numeric_data[idx], dtype=torch.float32)
        label_class = torch.tensor(self.labels_class[idx], dtype=torch.long)
        label_reg = torch.tensor(self.labels_reg[idx], dtype=torch.float32)
        return image, numeric, label_class, label_reg

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Shared Encoder
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        modules = list(base_model.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

    def forward(self, image, numeric):
        x_img = self.image_encoder(image).squeeze(-1).squeeze(-1)
        x_num = self.numeric_encoder(numeric)
        return torch.cat([x_img, x_num], dim=1)

# Stage 1 Classifier
class ClassifierHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(x)

# Stage 2 Regressor
class RegressorHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.regressor(x)

# Load Data
dataset = MultiModalDataset(CSV_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Stage 1: Train classifier
encoder = SharedEncoder().to(DEVICE)
classifier = ClassifierHead(512 + 64).to(DEVICE)
criterion_class = nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)

for epoch in range(NUM_EPOCHS_STAGE1):
    encoder.train()
    classifier.train()
    correct, total = 0, 0
    for imgs, nums, labels_class, _ in train_loader:
        imgs, nums, labels_class = imgs.to(DEVICE), nums.to(DEVICE), labels_class.to(DEVICE)
        features = encoder(imgs, nums)
        outputs = classifier(features)
        loss = criterion_class(outputs, labels_class)
        optimizer_class.zero_grad()
        loss.backward()
        optimizer_class.step()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels_class).sum().item()
        total += imgs.size(0)
    print(f"Stage 1 Epoch {epoch+1}, Train Acc: {correct/total:.4f}")

# Save encoder & classifier
torch.save(encoder.state_dict(), MODEL_CLASSIFIER_PATH)
torch.save(classifier.state_dict(), "classifier_head.pt")

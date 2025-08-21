import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Config
CSV_PATH = "/home/hujh/expriement/1-rearsh/LLaVA-main-c/merged_output.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS_STAGE2 = 30
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
        with torch.no_grad():
            x_img = self.image_encoder(image).squeeze(-1).squeeze(-1)
            x_num = self.numeric_encoder(numeric)
        return torch.cat([x_img, x_num], dim=1)

# Regressor Head
class RegressorHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.regressor(x)

# Accuracy-like metric

def acc_at_threshold(y_true, y_pred, threshold):
    return (np.abs(y_true - y_pred) < threshold).mean()

# Load Data
dataset = MultiModalDataset(CSV_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained encoder (frozen)
encoder = SharedEncoder().to(DEVICE)
encoder.load_state_dict(torch.load(MODEL_CLASSIFIER_PATH))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Train regressor
regressor = RegressorHead(512 + 64).to(DEVICE)
criterion_reg = nn.MSELoss()
optimizer_reg = torch.optim.Adam(regressor.parameters(), lr=1e-4)

for epoch in range(NUM_EPOCHS_STAGE2):
    regressor.train()
    total_loss, total_mae = 0, 0
    for imgs, nums, _, labels_reg in train_loader:
        imgs, nums, labels_reg = imgs.to(DEVICE), nums.to(DEVICE), labels_reg.to(DEVICE)
        features = encoder(imgs, nums)
        preds = regressor(features)
        loss = criterion_reg(preds, labels_reg)
        optimizer_reg.zero_grad()
        loss.backward()
        optimizer_reg.step()
        total_loss += loss.item() * imgs.size(0)
        total_mae += torch.abs(preds - labels_reg).sum().item()

    # Validation
    encoder.eval()
    regressor.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, nums, _, labels_reg in val_loader:
            imgs, nums = imgs.to(DEVICE), nums.to(DEVICE)
            features = encoder(imgs, nums)
            pred = regressor(features).cpu()
            y_true.append(labels_reg)
            y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Masked evaluation
    mask_c1 = y_true[:, 0] > 0
    mask_c2 = y_true[:, 1] > 0
    joint_mask = np.logical_and(mask_c1, mask_c2)

    mae = np.abs(y_true - y_pred).mean()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    acc_05_c1 = acc_at_threshold(y_true[mask_c1, 0], y_pred[mask_c1, 0], 0.5)
    acc_05_c2 = acc_at_threshold(y_true[mask_c2, 1], y_pred[mask_c2, 1], 0.5)
    acc_10_c1 = acc_at_threshold(y_true[mask_c1, 0], y_pred[mask_c1, 0], 1.0)
    acc_10_c2 = acc_at_threshold(y_true[mask_c2, 1], y_pred[mask_c2, 1], 1.0)

    acc_joint_05 = np.logical_and(
        np.abs(y_true[joint_mask, 0] - y_pred[joint_mask, 0]) < 0.5,
        np.abs(y_true[joint_mask, 1] - y_pred[joint_mask, 1]) < 0.5
    ).mean() if joint_mask.any() else 0.0

    acc_joint_10 = np.logical_and(
        np.abs(y_true[joint_mask, 0] - y_pred[joint_mask, 0]) < 1.0,
        np.abs(y_true[joint_mask, 1] - y_pred[joint_mask, 1]) < 1.0
    ).mean() if joint_mask.any() else 0.0

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_ds):.4f}, MAE: {total_mae/(len(train_ds)*2):.4f}")
    print(f"Val-MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Acc@0.5: conc1={acc_05_c1:.2%}, conc2={acc_05_c2:.2%}, joint={acc_joint_05:.2%}")
    print(f"Acc@1.0: conc1={acc_10_c1:.2%}, conc2={acc_10_c2:.2%}, joint={acc_joint_10:.2%}\n")

# Save model
torch.save(regressor.state_dict(), MODEL_REGRESSOR_PATH)
print("Regressor training complete. Saved.")


import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms  # Import torchvision transforms
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class FitzpatrickDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.data = df
        self.root_dir = root_dir
        self.transform = transform
        # Define a mapping from string labels to integer labels
        self.label_mapping = {'non-neoplastic': 0, 'benign': 1, 'malignant': 2} # Replace with your actual string labels

        # It's good practice to log the label mapping for verification
        logging.info(f"Label Mapping: {self.label_mapping}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['md5hash'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label_str = self.data.iloc[idx]['three_partition_label']

        # Convert the string label to an integer using the mapping
        if label_str in self.label_mapping:
            label = self.label_mapping[label_str]
        else:
            # Handle cases where the label is not in the mapping (optional)
            raise ValueError(f"Label '{label_str}' not found in the label mapping.")

        bias = int(self.data.iloc[idx]['fitzpatrick']) # Adjust to 0-indexed classes

        if self.transform:
            image = self.transform(image) # Apply the transform directly to the PIL Image

        # Convert the integer label to a PyTorch tensor of type long
        label = torch.tensor(label, dtype=torch.long)

        return image, label, bias

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/InvDiff/Ankur_InvClass_Fitz/data/finalfitz17k/'
TRAIN_METADATA_CSV = "/home/dse316/back/new/data/csv/fitzpatrick17k.csv"
LOG_FILE = 'Fitzpatrick17k/Final/training_efficientnetb3_fitzpatrick17k.log'
MODEL_SAVE_DIR = 'Fitzpatrick17k/Final/saved_models'
BEST_MODEL_NAME = 'efficientnetb3_fitzpatrick17k_best.pth'

# Hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
NUM_CLASSES = 3  # Assuming 3 classes for 'three_partition_label'
VALIDATION_INTERVAL = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SIZE = 0.2  # Define the validation set size (e.g., 20%)
RANDOM_SEED = 42  # For reproducibility

# --- Setup ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

logger.info(f"Using device: {DEVICE}")
logger.info("--- Configuration ---")
logger.info(f"Data Root Directory: {DATA_ROOT_DIR}")
logger.info(f"Train Metadata: {TRAIN_METADATA_CSV}")
#logger.info(f"Validation Metadata: {VAL_METADATA_CSV}")
logger.info(f"Log File: {LOG_FILE}")
logger.info(f"Model Save Directory: {MODEL_SAVE_DIR}")
logger.info(f"Best Model Name: {BEST_MODEL_NAME}")
logger.info(f"Number of Epochs: {NUM_EPOCHS}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Learning Rate: {LEARNING_RATE}")
logger.info(f"Number of Classes: {NUM_CLASSES}")
logger.info(f"Validation Interval: {VALIDATION_INTERVAL}")
logger.info("--- Starting Training ---")

# --- Data Loading ---
logger.info("Loading and splitting dataset...")

full_dataset = pd.read_csv(TRAIN_METADATA_CSV)  # Load the full dataset

# Split into train and validation
train_df, val_df = train_test_split(
    full_dataset,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=full_dataset['three_partition_label']  # Stratify by labels
)

# Define your transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

train_dataset = FitzpatrickDataset(
    df=train_df,
    root_dir=DATA_ROOT_DIR,
    transform=train_transforms  # Apply training transforms
)
val_dataset = FitzpatrickDataset(
    df=val_df,
    root_dir=DATA_ROOT_DIR,
    transform=val_transforms  # Apply validation transforms
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Training dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(val_dataset)}")

# --- Model Definition ---
logger.info("Initializing EfficientNetB3 model (from scratch)...")
model = efficientnet_b3(weights=None, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()  # You can add weights if needed for class imbalance
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
best_val_accuracy = 0.0
best_model_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    logger.info(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", unit="batch")
    for i, (images, labels, _) in enumerate(train_progress):
        # print(f"Type of images in DataLoader: {type(images)}")
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update progress bar
        train_progress.set_postfix(loss=(running_loss / total_train), acc=(100. * correct_train / total_train))

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = 100. * correct_train / len(train_dataset)
    logger.info(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")

    # --- Validation Phase ---
    if (epoch + 1) % VALIDATION_INTERVAL == 0:
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", unit="batch")
        with torch.no_grad():
            for images, labels, _ in val_progress:  # Assuming your dataset returns image, label, bias
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Update progress bar
                val_progress.set_postfix(loss=(running_val_loss / total_val), acc=(100. * correct_val / total_val))

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = 100. * correct_val / len(val_dataset)
        logger.info(f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save the best model
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"*** New best model saved with Val Acc: {best_val_accuracy:.2f}% to {best_model_path} ***")
        else:
            logger.info(
                f"Validation accuracy ({epoch_val_acc:.2f}%) did not improve from best ({best_val_accuracy:.2f}%).")

    epoch_duration = time.time() - start_time
    logger.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")

logger.info("--- Training Finished ---")
logger.info(f"Best Validation Accuracy achieved: {best_val_accuracy:.2f}%")
logger.info(f"Best model saved to: {best_model_path}")
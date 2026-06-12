# train_efficientnet.py
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from tqdm import tqdm
import pandas as pd

# Import your custom dataset
from ISIC_isic2 import ISICDataset # Make sure isic2.py is in src/data/

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/new/2019' # <<< CHANGE THIS
TRAIN_METADATA_CSV = os.path.join(DATA_ROOT_DIR, 'ISIC_2019_Train_Metadata_with_Disease.csv')
VAL_METADATA_CSV = os.path.join(DATA_ROOT_DIR, 'ISIC_2019_Test_Metadata_with_Disease.csv') # Using test set for validation

LOG_FILE = 'ISIC_effnet/ISIC_training_efficientnetb3_vanilla.log'
MODEL_SAVE_DIR = 'ISIC_effnet/ISIC_saved_models'
BEST_MODEL_NAME = 'efficientnetb3_isic_best.pth'

# Hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
NUM_CLASSES = 8 # Based on your ISICDataset's disease_mapping
VALIDATION_INTERVAL = 2 # Evaluate on validation set every 2 epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler() # Also print logs to console
                    ])
logger = logging.getLogger(__name__)

logger.info(f"Using device: {DEVICE}")
logger.info("--- Configuration ---")
logger.info(f"Data Root Directory: {DATA_ROOT_DIR}")
logger.info(f"Train Metadata: {TRAIN_METADATA_CSV}")
logger.info(f"Validation Metadata: {VAL_METADATA_CSV}")
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
logger.info("Loading datasets...")
# Use default transforms from ISICDataset
train_dataset = ISICDataset(root_dir=DATA_ROOT_DIR, split='train', metadata_csv=TRAIN_METADATA_CSV)
val_dataset = ISICDataset(root_dir=DATA_ROOT_DIR, split='val', metadata_csv=VAL_METADATA_CSV) # Use 'val' or 'test' split

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
logger.info(f"Training dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(val_dataset)}")

# --- Model Definition ---
logger.info("Initializing EfficientNetB3 model (from scratch)...")
# Load EfficientNetB3 without pre-trained weights
model = efficientnet_b3(weights=None, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW often works well

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
    for i, (images, labels, _) in enumerate(train_progress): # Ignore metadata for now
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
            for images, labels, _ in val_progress:
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
            logger.info(f"*** New best model saved with Val Acc: {best_val_accuracy:.2f}% to {best_model_path} ***")
        else:
            logger.info(f"Validation accuracy ({epoch_val_acc:.2f}%) did not improve from best ({best_val_accuracy:.2f}%).")

    epoch_duration = time.time() - start_time
    logger.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")


logger.info("--- Training Finished ---")
logger.info(f"Best Validation Accuracy achieved: {best_val_accuracy:.2f}%")
logger.info(f"Best model saved to: {best_model_path}")
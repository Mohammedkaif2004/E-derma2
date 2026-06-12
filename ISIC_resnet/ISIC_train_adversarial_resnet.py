# ISIC_train_adversarial_resnet.py
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights # Import necessary torchvision modules
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pandas as pd
import numpy as np
# Import custom modules
from isic3 import ISICDataset # Make sure src/data/isic3.py is updated
from grl import GradientReversalLayer # Make sure src/model/grl.py exists

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/new/2019' # <<< CHANGE THIS
TRAIN_METADATA_CSV = os.path.join(DATA_ROOT_DIR, 'ISIC_2019_Train_Metadata_with_Disease.csv')
VAL_METADATA_CSV = os.path.join(DATA_ROOT_DIR, 'ISIC_2019_Test_Metadata_with_Disease.csv') # Using test for validation

LOG_FILE = 'ISIC_resnet/training_adversarial_resnet18.log' # Changed log file name
MODEL_SAVE_DIR = 'ISIC_resnet/saved_models_adversarial'
BEST_MODEL_NAME = 'adversarial_resnet18_isic_best.pth' # Changed model name
CHECKPOINT_NAME = 'adversarial_resnet18_isic_checkpoint.pth' # Changed model name

# Hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 512 # Using the batch size from the user log
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_INTERVAL = 1
GRADIENT_CLIP_VAL = 1.0 # Optional: Gradient clipping

# Adversarial Setup
ADVERSARIAL_WEIGHT = 0.5
GRL_LAMBDA = 1.0
NUM_DISEASE_CLASSES = 8 # From ISICDataset
NUM_FITZPATRICK_CLASSES = 6 # 1-6 mapped to 0-5
MLP_HIDDEN_DIM = 512

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
logger.info("--- Adversarial Training Configuration (Skipping UNK Diseases) ---") # Updated Log Message
logger.info(f"Data Root Directory: {DATA_ROOT_DIR}")
logger.info(f"Log File: {LOG_FILE}")
logger.info(f"Model Save Directory: {MODEL_SAVE_DIR}")
logger.info(f"Number of Epochs: {NUM_EPOCHS}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Learning Rate: {LEARNING_RATE}")
logger.info(f"Weight Decay: {WEIGHT_DECAY}")
logger.info(f"Adversarial Weight (Loss Lambda): {ADVERSARIAL_WEIGHT}")
logger.info(f"GRL Lambda (Gradient Scaling): {GRL_LAMBDA}")
logger.info(f"MLP Hidden Dimension: {MLP_HIDDEN_DIM}")
logger.info(f"Number of Disease Classes: {NUM_DISEASE_CLASSES}")
logger.info(f"Number of Fitzpatrick Classes: {NUM_FITZPATRICK_CLASSES}")
logger.info(f"Validation Interval: {VALIDATION_INTERVAL}")
logger.info(f"Gradient Clipping: {GRADIENT_CLIP_VAL if GRADIENT_CLIP_VAL else 'Disabled'}")
logger.info("--- Starting Adversarial Training (Skipping UNK Diseases) ---") # Updated Log Message

# --- Data Loading ---
logger.info("Loading datasets...")
# Ensure ISICDataset maps 'UNK' disease to -1 and handles missing fitzpatrick (-1)
train_dataset = ISICDataset(root_dir=DATA_ROOT_DIR, split='train', metadata_csv=TRAIN_METADATA_CSV, handle_unknown_labels='exclude')
val_dataset = ISICDataset(root_dir=DATA_ROOT_DIR, split='val', metadata_csv=VAL_METADATA_CSV, handle_unknown_labels='exclude')

logger.info("Class weights are disabled for loss functions.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
logger.info(f"Training dataset size (before filtering UNK in batches): {len(train_dataset)}")
logger.info(f"Validation dataset size (before filtering UNK in batches): {len(val_dataset)}")
logger.info(f"Using {train_dataset.num_disease_classes} disease classes.")
logger.info(f"Using {train_dataset.num_fitzpatrick_classes} Fitzpatrick classes (0-5). Missing values coded as -1.")

# --- Model Definition ---
class MLPHead(nn.Module):
    """ 3-Layer MLP + Classification Layer """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim) # Classification layer

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class AdversarialDebiasingModel(nn.Module):
    def __init__(self, num_disease_classes, num_fitzpatrick_classes, mlp_hidden_dim, grl_lambda=1.0):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder_output_features = self.encoder.fc.in_features # Changed to ResNet18's fc layer
        self.encoder.fc = nn.Identity() # Changed to ResNet18's fc layer
        self.disease_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_disease_classes)
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.fitzpatrick_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_fitzpatrick_classes)

    def forward(self, x):
        features = self.encoder(x)
        disease_logits = self.disease_classifier(features)
        reversed_features = self.grl(features)
        fitzpatrick_logits = self.fitzpatrick_classifier(reversed_features)
        return disease_logits, fitzpatrick_logits

logger.info("Initializing Adversarial Debiasing Model...")
model = AdversarialDebiasingModel(
    num_disease_classes=NUM_DISEASE_CLASSES,
    num_fitzpatrick_classes=NUM_FITZPATRICK_CLASSES,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    grl_lambda=GRL_LAMBDA
).to(DEVICE)

# --- Loss Functions (Without Class Weights) ---
# Disease criterion does not need ignore_index as we filter -1 beforehand
disease_criterion = nn.CrossEntropyLoss()
# Fitzpatrick criterion still needs ignore_index for missing Fitz values (-1)
fitzpatrick_criterion = nn.CrossEntropyLoss(ignore_index=-1)

# --- Optimizer ---
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# --- Training Loop ---
best_val_disease_accuracy = 0.0
start_epoch = 0
best_model_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)
checkpoint_path = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_NAME)

# --- Resume Training (Optional) ---
# if os.path.exists(checkpoint_path):
#     logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     best_val_disease_accuracy = checkpoint['best_val_disease_accuracy']
#     if 'scheduler_state_dict' in checkpoint and scheduler is not None:
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     logger.info(f"Resuming from Epoch {start_epoch}, Best Val Disease Acc: {best_val_disease_accuracy:.2f}%")

for epoch in range(start_epoch, NUM_EPOCHS):
    start_time = time.time()
    logger.info(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} --- LR: {optimizer.param_groups[0]['lr']:.2e}")

    # --- Training Phase ---
    model.train()
    total_loss_train = 0.0
    disease_loss_train = 0.0
    fitzpatrick_loss_train = 0.0
    correct_disease_train = 0
    correct_fitzpatrick_train = 0
    total_samples_processed_train = 0 # Track samples *after* filtering UNK
    total_fitzpatrick_samples_train = 0 # Track Fitz samples *after* filtering UNK

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", unit="batch")
    # Use enumerate to get batch index 'i' for logging skipped batches
    for i, (images, disease_labels, fitzpatrick_labels) in enumerate(train_progress):
        images = images.to(DEVICE)
        disease_labels = disease_labels.to(DEVICE)
        fitzpatrick_labels = fitzpatrick_labels.to(DEVICE)

        # --- Filter out samples with unknown disease type (mapped to -1) ---
        valid_disease_mask = disease_labels != -1
        if not valid_disease_mask.any():
            # logger.warning(f"Skipping training batch {i} as all samples have unknown disease types.")
            continue # Skip to the next batch if no valid samples exist

        # Keep only valid samples
        images = images[valid_disease_mask]
        disease_labels = disease_labels[valid_disease_mask]
        fitzpatrick_labels = fitzpatrick_labels[valid_disease_mask]
        # --------------------------------------------------------------------

        current_batch_size = images.size(0) # Size after filtering

        optimizer.zero_grad()
        disease_logits, fitzpatrick_logits = model(images)

        # Calculate losses (without weights) on filtered data
        loss_d = disease_criterion(disease_logits, disease_labels)

        # Calculate Fitzpatrick loss only for samples with valid Fitz labels (already filtered for valid disease)
        valid_fitz_mask = fitzpatrick_labels != -1
        if valid_fitz_mask.sum() > 0:
            loss_f = fitzpatrick_criterion(
                fitzpatrick_logits[valid_fitz_mask],
                fitzpatrick_labels[valid_fitz_mask]
            )
        else:
            loss_f = torch.tensor(0.0, device=DEVICE, requires_grad=True) # Ensure it's a tensor on the correct device

        total_loss = loss_d + ADVERSARIAL_WEIGHT * loss_f
        total_loss.backward()

        if GRADIENT_CLIP_VAL:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)

        optimizer.step()

        # --- Statistics (use current_batch_size and counts based on filtered data) ---
        total_samples_processed_train += current_batch_size
        disease_loss_train += loss_d.item() * current_batch_size
        fitz_valid_count_in_batch = valid_fitz_mask.sum().item()
        if fitz_valid_count_in_batch > 0 :
            fitzpatrick_loss_train += loss_f.item() * fitz_valid_count_in_batch

        _, predicted_disease = torch.max(disease_logits.data, 1)
        correct_disease_train += (predicted_disease == disease_labels).sum().item()

        if fitz_valid_count_in_batch > 0:
            _, predicted_fitzpatrick = torch.max(fitzpatrick_logits[valid_fitz_mask].data, 1)
            correct_fitzpatrick_train += (predicted_fitzpatrick == fitzpatrick_labels[valid_fitz_mask]).sum().item()
            total_fitzpatrick_samples_train += fitz_valid_count_in_batch

        # Update progress bar - ensure denominators are not zero
        acc_d_epoch = (100. * correct_disease_train / total_samples_processed_train) if total_samples_processed_train > 0 else 0.0
        acc_f_epoch = (100. * correct_fitzpatrick_train / total_fitzpatrick_samples_train) if total_fitzpatrick_samples_train > 0 else 0.0
        train_progress.set_postfix(
            loss=total_loss.item(), # Instantaneous loss
            loss_d=loss_d.item(),
            loss_f=loss_f.item(),
            acc_d=acc_d_epoch, # Epoch accuracy
            acc_f=acc_f_epoch  # Epoch accuracy
        )

    # Calculate average losses and accuracies for the epoch based on processed samples
    avg_train_loss = total_loss_train / total_samples_processed_train if total_samples_processed_train > 0 else 0.0
    avg_disease_loss_train = disease_loss_train / total_samples_processed_train if total_samples_processed_train > 0 else 0.0
    avg_fitzpatrick_loss_train = fitzpatrick_loss_train / total_fitzpatrick_samples_train if total_fitzpatrick_samples_train > 0 else 0.0
    # Accuracies already calculated correctly using totals
    train_disease_acc = acc_d_epoch
    train_fitzpatrick_acc = acc_f_epoch

    logger.info(f"Epoch {epoch+1} Train Summary: Processed Samples: {total_samples_processed_train} | Avg Loss: {avg_train_loss:.4f} | Disease Loss: {avg_disease_loss_train:.4f} | Fitz Loss: {avg_fitzpatrick_loss_train:.4f}")
    logger.info(f"Epoch {epoch+1} Train Accuracy: Disease: {train_disease_acc:.2f}% | Fitzpatrick: {train_fitzpatrick_acc:.2f}% ({total_fitzpatrick_samples_train} valid Fitz samples)")


    # --- Validation Phase ---
    if (epoch + 1) % VALIDATION_INTERVAL == 0:
        model.eval()
        total_loss_val = 0.0
        disease_loss_val = 0.0
        fitzpatrick_loss_val = 0.0
        correct_disease_val = 0
        correct_fitzpatrick_val = 0
        total_samples_processed_val = 0 # Track samples *after* filtering UNK
        total_fitzpatrick_samples_val = 0 # Track Fitz samples *after* filtering UNK

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", unit="batch")
        # Use enumerate to get batch index 'i' for logging skipped batches
        with torch.no_grad():
            for i, (images, disease_labels, fitzpatrick_labels) in enumerate(val_progress):
                images = images.to(DEVICE)
                disease_labels = disease_labels.to(DEVICE)
                fitzpatrick_labels = fitzpatrick_labels.to(DEVICE)

                # --- Filter out samples with unknown disease type (mapped to -1) ---
                valid_disease_mask = disease_labels != -1
                if not valid_disease_mask.any():
                    # logger.warning(f"Skipping validation batch {i} as all samples have unknown disease types.")
                    continue # Skip to the next batch

                images = images[valid_disease_mask]
                disease_labels = disease_labels[valid_disease_mask]
                fitzpatrick_labels = fitzpatrick_labels[valid_disease_mask]
                # --------------------------------------------------------------------

                current_batch_size = images.size(0) # Size after filtering

                disease_logits, fitzpatrick_logits = model(images)

                # Calculate losses (without weights)
                loss_d = disease_criterion(disease_logits, disease_labels)
                valid_fitz_mask = fitzpatrick_labels != -1 # Re-calculate for Fitz on the filtered batch
                if valid_fitz_mask.sum() > 0:
                    loss_f = fitzpatrick_criterion(
                        fitzpatrick_logits[valid_fitz_mask],
                        fitzpatrick_labels[valid_fitz_mask]
                    )
                else:
                    loss_f = torch.tensor(0.0, device=DEVICE)

                total_loss = loss_d + ADVERSARIAL_WEIGHT * loss_f

                # Statistics
                total_samples_processed_val += current_batch_size
                disease_loss_val += loss_d.item() * current_batch_size
                fitz_valid_count_in_batch = valid_fitz_mask.sum().item()
                if fitz_valid_count_in_batch > 0:
                    fitzpatrick_loss_val += loss_f.item() * fitz_valid_count_in_batch

                _, predicted_disease = torch.max(disease_logits.data, 1)
                correct_disease_val += (predicted_disease == disease_labels).sum().item()

                if fitz_valid_count_in_batch > 0:
                    _, predicted_fitzpatrick = torch.max(fitzpatrick_logits[valid_fitz_mask].data, 1)
                    correct_fitzpatrick_val += (predicted_fitzpatrick == fitzpatrick_labels[valid_fitz_mask]).sum().item()
                    total_fitzpatrick_samples_val += fitz_valid_count_in_batch

                # Update progress bar using epoch totals for accuracy
                acc_d_epoch_val = (100. * correct_disease_val / total_samples_processed_val) if total_samples_processed_val > 0 else 0.0
                acc_f_epoch_val = (100. * correct_fitzpatrick_val / total_fitzpatrick_samples_val) if total_fitzpatrick_samples_val > 0 else 0.0
                val_progress.set_postfix(
                    loss=total_loss.item(),
                    acc_d=acc_d_epoch_val, # Epoch accuracy
                    acc_f=acc_f_epoch_val  # Epoch accuracy
                )

        # Calculate average losses and accuracies for validation
        avg_val_loss = total_loss_val / total_samples_processed_val if total_samples_processed_val > 0 else 0.0
        avg_disease_loss_val = disease_loss_val / total_samples_processed_val if total_samples_processed_val > 0 else 0.0
        avg_fitzpatrick_loss_val = fitzpatrick_loss_val / total_fitzpatrick_samples_val if total_fitzpatrick_samples_val > 0 else 0.0
        val_disease_acc = acc_d_epoch_val
        val_fitzpatrick_acc = acc_f_epoch_val

        logger.info(f"Epoch {epoch+1} Val Summary: Processed Samples: {total_samples_processed_val} | Avg Loss: {avg_val_loss:.4f} | Disease Loss: {avg_disease_loss_val:.4f} | Fitz Loss: {avg_fitzpatrick_loss_val:.4f}")
        logger.info(f"Epoch {epoch+1} Val Accuracy: Disease: {val_disease_acc:.2f}% | Fitzpatrick: {val_fitzpatrick_acc:.2f}% ({total_fitzpatrick_samples_val} valid Fitz samples)")


        # --- Save Best Model based on Disease Validation Accuracy ---
        if val_disease_acc > best_val_disease_accuracy:
            best_val_disease_accuracy = val_disease_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"*** New best model saved based on Disease Val Acc: {best_val_disease_accuracy:.2f}% to {best_model_path} ***")
        else:
            logger.info(f"Validation disease accuracy ({val_disease_acc:.2f}%) did not improve from best ({best_val_disease_accuracy:.2f}%).")

        # --- Save Checkpoint ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_disease_accuracy': best_val_disease_accuracy,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # --- Learning Rate Scheduler Step ---
        if scheduler:
            # Step based on the primary metric: disease validation accuracy
            scheduler.step(val_disease_acc)


    epoch_duration = time.time() - start_time
    logger.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")


logger.info("--- Adversarial Training Finished (Skipped UNK Diseases) ---") # Updated Log Message
logger.info(f"Best Validation Disease Accuracy achieved: {best_val_disease_accuracy:.2f}%")
logger.info(f"Best model saved to: {best_model_path}")
# FITZPATRICK17K_train_adversarial_resnet.py
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights  # Changed import
from torchvision import transforms  # Import torchvision transforms
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Import custom modules (assuming these are copied to the same directory or path is adjusted)
class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).

    This layer reverses the sign of the gradient during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambda_, None)

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)

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

class FitzpatrickDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.data = df
        self.root_dir = root_dir
        self.transform = transform
        # Define a mapping from string labels to integer labels
        self.label_mapping = {'non-neoplastic': 0, 'benign': 1, 'malignant': 2} # Replace with your actual string labels
        self.fitzpatrick_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5} # Map Fitzpatrick 1-6 to 0-5

        # It's good practice to log the label mapping for verification
        logging.info(f"Disease Label Mapping: {self.label_mapping}")
        logging.info(f"Fitzpatrick Label Mapping: {self.fitzpatrick_mapping}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['md5hash'] + '.jpg')
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            logging.error(f"Image not found: {img_name}")
            raise
        label_str = self.data.iloc[idx]['three_partition_label']

        if label_str in self.label_mapping:
            label = self.label_mapping[label_str]
        else:
            raise ValueError(f"Disease label '{label_str}' not found in the label mapping.")

        fitzpatrick_raw = self.data.iloc[idx]['fitzpatrick']
        try:
            bias_label = int(fitzpatrick_raw) - 1
            if not (0 <= bias_label <= 5):
                bias_label = -1
        except (ValueError, TypeError):
            bias_label = -1

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        bias = torch.tensor(bias_label, dtype=torch.long)

        return image, label, bias

class AdversarialDebiasingModel(nn.Module):
    def __init__(self, num_disease_classes, num_fitzpatrick_classes, mlp_hidden_dim, grl_lambda=1.0):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder_output_features = self.encoder.fc.in_features # Changed to ResNet18's fc layer
        self.encoder.fc = nn.Identity() # Remove the original fully connected layer
        self.disease_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_disease_classes)
        self.grl = GRL(lambda_=grl_lambda)
        self.fitzpatrick_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_fitzpatrick_classes)

    def forward(self, x):
        features = self.encoder(x)
        disease_logits = self.disease_classifier(features)
        reversed_features = self.grl(features)
        fitzpatrick_logits = self.fitzpatrick_classifier(reversed_features)
        return disease_logits, fitzpatrick_logits

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/InvDiff/Ankur_InvClass_Fitz/data/finalfitz17k/'
TRAIN_METADATA_CSV = "/home/dse316/back/new/data/csv/fitzpatrick17k.csv"
LOG_FILE = 'Fitzpatrick17k/Final/training_adversarial_resnet18_fitzpatrick17k.log'  # Renamed log file
MODEL_SAVE_DIR = 'Fitzpatrick17k/Final/saved_models_adversarial_fitz_resnet'
BEST_MODEL_NAME = 'adversarial_resnet18_fitzpatrick17k_best.pth'  # Renamed best model name
CHECKPOINT_NAME = 'adversarial_resnet18_fitzpatrick17k_checkpoint.pth' # Renamed checkpoint name

# Hyperparameters
NUM_EPOCHS = 4 # Adjusted number of epochs
BATCH_SIZE = 512 # Using the batch size from the ISIC code
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_DISEASE_CLASSES = 3  # Assuming 3 classes for 'three_partition_label'
NUM_FITZPATRICK_CLASSES = 6 # 1-6 mapped to 0-5
VALIDATION_INTERVAL = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SIZE = 0.2  # Define the validation set size (e.g., 20%)
RANDOM_SEED = 42  # For reproducibility
GRADIENT_CLIP_VAL = 1.0 # Optional: Gradient clipping

# Adversarial Setup
ADVERSARIAL_WEIGHT = 0.5
GRL_LAMBDA = 1.0
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
logger.info("--- Adversarial Training Configuration ---")
logger.info(f"Data Root Directory: {DATA_ROOT_DIR}")
logger.info(f"Train Metadata: {TRAIN_METADATA_CSV}")
logger.info(f"Log File: {LOG_FILE}")
logger.info(f"Model Save Directory: {MODEL_SAVE_DIR}")
logger.info(f"Best Model Name: {BEST_MODEL_NAME}")
logger.info(f"Checkpoint Name: {CHECKPOINT_NAME}")
logger.info(f"Number of Epochs: {NUM_EPOCHS}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Learning Rate: {LEARNING_RATE}")
logger.info(f"Weight Decay: {WEIGHT_DECAY}")
logger.info(f"Number of Disease Classes: {NUM_DISEASE_CLASSES}")
logger.info(f"Number of Fitzpatrick Classes: {NUM_FITZPATRICK_CLASSES}")
logger.info(f"Validation Interval: {VALIDATION_INTERVAL}")
logger.info(f"Adversarial Weight (Loss Lambda): {ADVERSARIAL_WEIGHT}")
logger.info(f"GRL Lambda (Gradient Scaling): {GRL_LAMBDA}")
logger.info(f"MLP Hidden Dimension: {MLP_HIDDEN_DIM}")
logger.info(f"Gradient Clipping: {GRADIENT_CLIP_VAL if GRADIENT_CLIP_VAL else 'Disabled'}")
logger.info("--- Starting Adversarial Training ---")

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

# Define your transformations (using ISIC transforms for consistency, can be adjusted)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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
logger.info("Initializing Adversarial Debiasing Model...")
model = AdversarialDebiasingModel(
    num_disease_classes=NUM_DISEASE_CLASSES,
    num_fitzpatrick_classes=NUM_FITZPATRICK_CLASSES,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    grl_lambda=GRL_LAMBDA
).to(DEVICE)

# --- Loss Functions ---
disease_criterion = nn.CrossEntropyLoss()
fitzpatrick_criterion = nn.CrossEntropyLoss(ignore_index=-1) # Ignore missing Fitzpatrick values

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
    total_samples_processed_train = 0
    total_fitzpatrick_samples_train = 0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", unit="batch")
    for i, (images, disease_labels, fitzpatrick_labels) in enumerate(train_progress):
        images = images.to(DEVICE)
        disease_labels = disease_labels.to(DEVICE)
        fitzpatrick_labels = fitzpatrick_labels.to(DEVICE)

        optimizer.zero_grad()
        disease_logits, fitzpatrick_logits = model(images)

        loss_d = disease_criterion(disease_logits, disease_labels)
        loss_f = fitzpatrick_criterion(fitzpatrick_logits, fitzpatrick_labels)

        total_loss = loss_d + ADVERSARIAL_WEIGHT * loss_f
        total_loss.backward()

        if GRADIENT_CLIP_VAL:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)

        optimizer.step()

        # --- Statistics ---
        total_samples_processed_train += images.size(0)
        disease_loss_train += loss_d.item() * images.size(0)
        fitzpatrick_loss_train += loss_f.item() * images.size(0)

        _, predicted_disease = torch.max(disease_logits.data, 1)
        correct_disease_train += (predicted_disease == disease_labels).sum().item()

        # Calculate Fitzpatrick accuracy only for non-missing labels
        valid_fitz_mask = fitzpatrick_labels != -1
        if valid_fitz_mask.sum() > 0:
            _, predicted_fitzpatrick = torch.max(fitzpatrick_logits[valid_fitz_mask].data, 1)
            correct_fitzpatrick_train += (predicted_fitzpatrick == fitzpatrick_labels[valid_fitz_mask]).sum().item()
            total_fitzpatrick_samples_train += valid_fitz_mask.sum().item()

        # Update progress bar
        acc_d_epoch = (100. * correct_disease_train / total_samples_processed_train) if total_samples_processed_train > 0 else 0.0
        acc_f_epoch = (100. * correct_fitzpatrick_train / total_fitzpatrick_samples_train) if total_fitzpatrick_samples_train > 0 else 0.0
        train_progress.set_postfix(
            loss=total_loss.item(),
            loss_d=loss_d.item(),
            loss_f=loss_f.item(),
            acc_d=acc_d_epoch,
            acc_f=acc_f_epoch
        )

    avg_train_loss = total_loss_train / total_samples_processed_train if total_samples_processed_train > 0 else 0.0
    avg_disease_loss_train = disease_loss_train / total_samples_processed_train if total_samples_processed_train > 0 else 0.0
    avg_fitzpatrick_loss_train = fitzpatrick_loss_train / total_samples_processed_train if total_samples_processed_train > 0 else 0.0
    train_disease_acc = acc_d_epoch
    train_fitzpatrick_acc = acc_f_epoch

    logger.info(f"Epoch {epoch+1} Train Summary: Avg Loss: {avg_train_loss:.4f} | Disease Loss: {avg_disease_loss_train:.4f} | Fitz Loss: {avg_fitzpatrick_loss_train:.4f}")
    logger.info(f"Epoch {epoch+1} Train Accuracy: Disease: {train_disease_acc:.2f}% | Fitzpatrick: {train_fitzpatrick_acc:.2f}% ({total_fitzpatrick_samples_train} valid Fitz samples)")

    # --- Validation Phase ---
    if (epoch + 1) % VALIDATION_INTERVAL == 0:
        model.eval()
        total_loss_val = 0.0
        disease_loss_val = 0.0
        fitzpatrick_loss_val = 0.0
        correct_disease_val = 0
        correct_fitzpatrick_val = 0
        total_samples_processed_val = 0
        total_fitzpatrick_samples_val = 0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", unit="batch")
        with torch.no_grad():
            for images, disease_labels, fitzpatrick_labels in val_progress:
                images = images.to(DEVICE)
                disease_labels = disease_labels.to(DEVICE)
                fitzpatrick_labels = fitzpatrick_labels.to(DEVICE)

                disease_logits, fitzpatrick_logits = model(images)

                loss_d = disease_criterion(disease_logits, disease_labels)
                loss_f = fitzpatrick_criterion(fitzpatrick_logits, fitzpatrick_labels)

                total_loss = loss_d + ADVERSARIAL_WEIGHT * loss_f

                # Statistics
                total_samples_processed_val += images.size(0)
                disease_loss_val += loss_d.item() * images.size(0)
                fitzpatrick_loss_val += loss_f.item() * images.size(0)

                _, predicted_disease = torch.max(disease_logits.data, 1)
                correct_disease_val += (predicted_disease == disease_labels).sum().item()

                # Calculate Fitzpatrick accuracy only for non-missing labels
                valid_fitz_mask = fitzpatrick_labels != -1
                if valid_fitz_mask.sum() > 0:
                    _, predicted_fitzpatrick = torch.max(fitzpatrick_logits[valid_fitz_mask].data, 1)
                    correct_fitzpatrick_val += (predicted_fitzpatrick == fitzpatrick_labels[valid_fitz_mask]).sum().item()
                    total_fitzpatrick_samples_val += valid_fitz_mask.sum().item()

                # Update progress bar
                acc_d_epoch_val = (100. * correct_disease_val / total_samples_processed_val) if total_samples_processed_val > 0 else 0.0
                acc_f_epoch_val = (100. * correct_fitzpatrick_val / total_fitzpatrick_samples_val) if total_fitzpatrick_samples_val > 0 else 0.0
                val_progress.set_postfix(
                    loss=total_loss.item(),
                    acc_d=acc_d_epoch_val,
                    acc_f=acc_f_epoch_val
                )

        avg_val_loss = total_loss_val / total_samples_processed_val if total_samples_processed_val > 0 else 0.0
        avg_disease_loss_val = disease_loss_val / total_samples_processed_val if total_samples_processed_val > 0 else 0.0
        avg_fitzpatrick_loss_val = fitzpatrick_loss_val / total_samples_processed_val if total_samples_processed_val > 0 else 0.0
        val_disease_acc = acc_d_epoch_val
        val_fitzpatrick_acc = acc_f_epoch_val

        logger.info(f"Epoch {epoch+1} Val Summary: Avg Loss: {avg_val_loss:.4f} | Disease Loss: {avg_disease_loss_val:.4f} | Fitz Loss: {avg_fitzpatrick_loss_val:.4f}")
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
            scheduler.step(val_disease_acc)

    epoch_duration = time.time() - start_time
    logger.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")

logger.info("--- Adversarial Training Finished ---")
logger.info(f"Best Validation Disease Accuracy achieved: {best_val_disease_accuracy:.2f}%")
logger.info(f"Best model saved to: {best_model_path}")
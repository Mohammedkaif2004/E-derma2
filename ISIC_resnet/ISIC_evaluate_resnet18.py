import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18  # Changed model
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from collections import defaultdict

# Import your custom dataset
from isic3 import ISICDataset  # Changed import

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/new/2019'  # <<< CHANGE THIS
TEST_METADATA_CSV = os.path.join(DATA_ROOT_DIR,
                                 'ISIC_2019_Test_Metadata_with_Disease.csv')  # <<< Or your specific test CSV
# SAVED_MODEL_PATH = '/home/dse316/back/new/saved_models/efficientnetb3_isic_best.pth' # <<< Path to your trained model
SAVED_MODEL_PATH = "/home/dse316/back/InvDiff/FairAlign_Final_Results/mdkaif/ISIC_resnet/saved_models/resnet18_isic_best.pth"  # Changed model path
LOG_FILE = 'ISIC_resnet/evaluation_resnet18.log'  # Changed log file name

# Evaluation Parameters
BATCH_SIZE = 64  # Can often be larger for evaluation
NUM_CLASSES = 8  # Must match the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
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
logger.info(f"Test Metadata: {TEST_METADATA_CSV}")
logger.info(f"Saved Model Path: {SAVED_MODEL_PATH}")
logger.info(f"Log File: {LOG_FILE}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Number of Classes: {NUM_CLASSES}")
logger.info("--- Starting Evaluation ---")

# --- Data Loading ---
logger.info("Loading test dataset...")
# Use the same transforms as validation/training (without augmentation if used)
test_dataset = ISICDataset(root_dir=DATA_ROOT_DIR,
                                   split='test',
                                   metadata_csv=TEST_METADATA_CSV,
                                   handle_unknown_labels='exclude')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
logger.info(f"Test dataset size: {len(test_dataset)}")

# --- Model Definition and Loading Weights ---
logger.info("Initializing ResNet18 model architecture...")  # Changed log message
model = resnet18(weights=None, num_classes=NUM_CLASSES)  # Initialize the architecture # Changed model

logger.info(f"Loading trained weights from: {SAVED_MODEL_PATH}")
if not os.path.exists(SAVED_MODEL_PATH):
    logger.error(f"Model file not found at {SAVED_MODEL_PATH}")
    exit()
try:
    # Load state dict - use map_location for flexibility
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
    logger.info("Model weights loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model weights: {e}")
    exit()

model = model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# --- Evaluation Loop ---
all_preds = []
all_labels = []
all_fitzpatrick_scales = []
running_loss = 0.0
total_samples = 0
criterion = nn.CrossEntropyLoss()  # Optional: to calculate loss

logger.info("Starting evaluation loop...")
eval_progress = tqdm(test_loader, desc="Evaluating", unit="batch")
with torch.no_grad():  # Disable gradient calculations
    # --- Modified loop to capture metadata ---
    for images, labels, fitzpatrick_scales in eval_progress:
        # --- Ensure labels are within the valid range [0, NUM_CLASSES - 1] ---
        labels = torch.clamp(labels, min=0, max=NUM_CLASSES - 1)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)  # Calculate loss (optional)

        _, predicted = torch.max(outputs.data, 1)

        running_loss += loss.item() * images.size(0)
        total_samples += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # --- Store Fitzpatrick scale for each sample in the batch ---
        # Now 'fitzpatrick_scales' directly contains the Fitzpatrick scales for the batch
        # Ensure scales are numeric (convert tensor elements if necessary)
        numeric_scales = [s.item() if isinstance(s, torch.Tensor) else float(s) for s in
                          fitzpatrick_scales]
        all_fitzpatrick_scales.extend(numeric_scales)
        # --- End modification ---

        # Update progress bar description with running loss/accuracy
        # Calculate accuracy on the fly only if lists are not empty
        if all_labels:
            current_acc = accuracy_score(all_labels, all_preds) * 100
            eval_progress.set_postfix(loss=(running_loss / total_samples), acc=f"{current_acc:.2f}%")
        else:
            eval_progress.set_postfix(loss=(running_loss / total_samples), acc="N/A")

# --- Calculate and Log Metrics ---
logger.info("\n--- Evaluation Finished ---")

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_fitzpatrick_scales = np.array(all_fitzpatrick_scales)  # <--- Convert scales too

if len(all_labels) != len(all_fitzpatrick_scales):
    logger.warning(
        f"Mismatch in number of labels ({len(all_labels)}) and Fitzpatrick scales ({len(all_fitzpatrick_scales)}). "
        f"Check metadata loading.")
    # Continue but the per-scale accuracy might be wrong/incomplete

# Overall Accuracy
overall_accuracy = accuracy_score(all_labels, all_preds) * 100
final_loss = running_loss / len(test_dataset) if len(test_dataset) > 0 else 0
logger.info(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
logger.info(f"Average Test Loss: {final_loss:.4f}")

# Class names from your dataset mapping
class_names = list(test_dataset.disease_mapping.keys())
target_names = [name for name, idx in sorted(test_dataset.disease_mapping.items(),
                                             key=lambda item: item[1])]  # Ensure correct order

# Classification Report
logger.info("\nClassification Report:")
try:
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=3,
                                     zero_division=0)
    logger.info(f"\n{report}")
except ValueError as e:
    logger.error(f"Could not generate classification report: {e}")

# Confusion Matrix
logger.info("\nConfusion Matrix:")
try:
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    logger.info(f"\n{cm_df.to_string()}")
    # Optionally save confusion matrix to CSV
    cm_df.to_csv("evaluation_confusion_matrix.csv")
    logger.info("Confusion matrix saved to evaluation_confusion_matrix.csv")
except ValueError as e:
    logger.error(f"Could not generate confusion matrix: {e}")

# --- Fitzpatrick Scale Accuracy Analysis ---
logger.info("\n--- Accuracy per Fitzpatrick Scale ---")

unique_scales = np.unique(all_fitzpatrick_scales)
# Sort scales for consistent reporting, handle potential NaN or non-numeric if necessary
try:
    # Attempt numeric sort, placing NaN/-1 at the end if present
    numeric_unique_scales = np.sort(unique_scales[~np.isnan(unique_scales)])
    if np.isnan(unique_scales).any():
        numeric_unique_scales = np.append(numeric_unique_scales, np.nan)
    unique_scales = numeric_unique_scales
except TypeError:
    logger.warning("Could not sort Fitzpatrick scales numerically, using default order.")

fitzpatrick_results = {}

for scale in unique_scales:
    scale_label = "Unknown/Missing (-1)" if scale == -1 else f"Scale {int(scale)}"  # Handle -1 case from dataset
    if np.isnan(scale):
        scale_label = "NaN"
        indices = np.isnan(all_fitzpatrick_scales)
    else:
        indices = (all_fitzpatrick_scales == scale)

    if np.sum(indices) == 0:
        logger.info(f"{scale_label}: 0 samples.")
        continue  # Skip if no samples for this scale

    subset_labels = all_labels[indices]
    subset_preds = all_preds[indices]

    scale_accuracy = accuracy_score(subset_labels, subset_preds) * 100
    num_samples = len(subset_labels)

    logger.info(f"{scale_label}: Accuracy = {scale_accuracy:.2f}% ({num_samples} samples)")
    fitzpatrick_results[scale_label] = {'accuracy': scale_accuracy, 'count': num_samples}

# You can also save these results if needed
# results_df = pd.DataFrame.from_dict(fitzpatrick_results, orient='index')
# results_df.to_csv("evaluation_fitzpatrick_accuracy.csv")
# logger.info("Fitzpatrick scale accuracy saved to evaluation_fitzpatrick_accuracy.csv")

logger.info("\nEvaluation complete.")
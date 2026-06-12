# evaluate_efficientnet.py (standalone version for adversarial training)
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights # Import necessary torchvision modules
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Import your custom dataset
from ISIC_isic3 import ISICDataset

# --- Define the Model Architecture Here (Copied from train_adversarial.py) ---
class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer.

    This layer reverses the sign of the gradients during backpropagation.
    """
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.lambda_, None)

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
        self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        encoder_output_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.disease_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_disease_classes)
        self.grl = GradientReversalLayer.apply # Use the apply method for the static function
        self.grl_lambda = grl_lambda # Store lambda as an attribute
        self.fitzpatrick_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_fitzpatrick_classes)

    def forward(self, x):
        features = self.encoder(x)
        disease_logits = self.disease_classifier(features)
        reversed_features = self.grl(features, self.grl_lambda) # Pass lambda as an argument
        fitzpatrick_logits = self.fitzpatrick_classifier(reversed_features)
        return disease_logits, fitzpatrick_logits

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/new/2019' # <<< CHANGE THIS
TEST_METADATA_CSV = os.path.join(DATA_ROOT_DIR, 'ISIC_2019_Test_Metadata_with_Disease.csv') # <<< Or your specific test CSV
# SAVED_MODEL_PATH = '/home/dse316/back/new/saved_models/efficientnetb3_isic_best.pth' # <<< Path to your trained model
SAVED_MODEL_PATH ="/home/dse316/back/InvDiff/FairAlign_Final_Results/mdkaif/ISIC_effnet/ISIC_saved_models_adversarial/adversarial_effb3_isic_best.pth"
LOG_FILE = 'ISIC_effnet/ISIC_evaluation_adversarial_effb3.log' # Changed log file name

# Evaluation Parameters
BATCH_SIZE = 64 # Can often be larger for evaluation
NUM_DISEASE_CLASSES = 8 # Must match the trained model for disease classification
NUM_FITZPATRICK_CLASSES = 6 # Must match the trained model for Fitzpatrick classification
MLP_HIDDEN_DIM = 512 # Should match the training configuration
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
logger.info(f"Number of Disease Classes: {NUM_DISEASE_CLASSES}")
logger.info(f"Number of Fitzpatrick Classes: {NUM_FITZPATRICK_CLASSES}")
logger.info(f"MLP Hidden Dimension: {MLP_HIDDEN_DIM}")
logger.info("--- Starting Evaluation ---")

# --- Data Loading ---
logger.info("Loading test dataset...")
# Use the same transforms as validation/training (without augmentation if used)
test_dataset = ISICDataset(root_dir=DATA_ROOT_DIR, split='test', metadata_csv=TEST_METADATA_CSV, filter_missing_fitzpatrick=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
logger.info(f"Test dataset size: {len(test_dataset)}")

# --- Model Definition and Loading Weights ---
logger.info("Initializing AdversarialDebiasingModel architecture...")
model = AdversarialDebiasingModel(
    num_disease_classes=NUM_DISEASE_CLASSES,
    num_fitzpatrick_classes=NUM_FITZPATRICK_CLASSES,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    grl_lambda=1.0 # GRL lambda is used during training, set to a default value here
).to(DEVICE)

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

model.eval() # Set model to evaluation mode

# --- Evaluation Loop ---
all_disease_preds = []
all_disease_labels = []
all_fitzpatrick_scales = [] # <--- New list to store Fitzpatrick scales
running_loss = 0.0
total_samples = 0
disease_criterion = nn.CrossEntropyLoss() # Optional: to calculate loss

logger.info("Starting evaluation loop...")
eval_progress = tqdm(test_loader, desc="Evaluating", unit="batch")
with torch.no_grad(): # Disable gradient calculations
    # --- Modified loop to capture metadata ---
    for images, disease_labels, fitzpatrick_labels in eval_progress: # Directly unpack assuming order
        images = images.to(DEVICE)
        disease_labels = disease_labels.to(DEVICE)
        fitzpatrick_labels = fitzpatrick_labels.to(DEVICE)

        disease_logits, fitzpatrick_logits = model(images) # Get both outputs

        # Filter out samples with unknown disease type (-1) for evaluation
        valid_disease_mask = disease_labels != -1
        if not valid_disease_mask.any():
            continue

        valid_images = images[valid_disease_mask]
        valid_disease_labels = disease_labels[valid_disease_mask]
        valid_fitzpatrick_labels = fitzpatrick_labels[valid_disease_mask]

        outputs_disease = disease_logits[valid_disease_mask] # Use disease logits for evaluation
        loss = disease_criterion(outputs_disease, valid_disease_labels) # Calculate loss (optional)

        _, predicted_disease = torch.max(outputs_disease.data, 1)

        running_loss += loss.item() * valid_images.size(0)
        total_samples += valid_images.size(0)

        all_disease_preds.extend(predicted_disease.cpu().numpy())
        all_disease_labels.extend(valid_disease_labels.cpu().numpy())
        # --- Store Fitzpatrick scale for each sample in the batch ---
        # Now 'fitzpatrick_labels' directly contains the Fitzpatrick scales for the batch
        all_fitzpatrick_scales.extend(valid_fitzpatrick_labels.cpu().numpy())
        # --- End modification ---

        # Update progress bar description with running loss/accuracy
        # Calculate accuracy on the fly only if lists are not empty
        if all_disease_labels:
             current_acc = accuracy_score(all_disease_labels, all_disease_preds) * 100
             eval_progress.set_postfix(loss=(running_loss/total_samples), acc=f"{current_acc:.2f}%")
        else:
             eval_progress.set_postfix(loss=(running_loss/total_samples), acc="N/A")


# --- Calculate and Log Metrics ---
logger.info("\n--- Evaluation Finished ---")

# Convert lists to numpy arrays
all_disease_preds = np.array(all_disease_preds)
all_disease_labels = np.array(all_disease_labels)
all_fitzpatrick_scales = np.array(all_fitzpatrick_scales) # <--- Convert scales too

if len(all_disease_labels) != len(all_fitzpatrick_scales):
    logger.warning(f"Mismatch in number of disease labels ({len(all_disease_labels)}) and Fitzpatrick scales ({len(all_fitzpatrick_scales)}). Check metadata loading.")
    # Continue but the per-scale accuracy might be wrong/incomplete

# Overall Accuracy
overall_accuracy = accuracy_score(all_disease_labels, all_disease_preds) * 100
final_loss = running_loss / len(test_dataset) if len(test_dataset) > 0 else 0
logger.info(f"Overall Test Accuracy (Disease): {overall_accuracy:.2f}%")
logger.info(f"Average Test Loss (Disease): {final_loss:.4f}")

# Class names from your dataset mapping
class_names = list(test_dataset.disease_mapping.keys())
target_names = [name for name, idx in sorted(test_dataset.disease_mapping.items(), key=lambda item: item[1])] # Ensure correct order

# Classification Report
logger.info("\nClassification Report (Disease):")
try:
    report = classification_report(all_disease_labels, all_disease_preds, target_names=target_names, digits=3, zero_division=0)
    logger.info(f"\n{report}")
except ValueError as e:
    logger.error(f"Could not generate classification report: {e}")


# Confusion Matrix
logger.info("\nConfusion Matrix (Disease):")
try:
    cm = confusion_matrix(all_disease_labels, all_disease_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    logger.info(f"\n{cm_df.to_string()}")
    # Optionally save confusion matrix to CSV
    cm_df.to_csv("evaluation_confusion_matrix_adversarial.csv")
    logger.info("Confusion matrix saved to evaluation_confusion_matrix_adversarial.csv")
except ValueError as e:
     logger.error(f"Could not generate confusion matrix: {e}")


# --- Fitzpatrick Scale Accuracy Analysis ---
logger.info("\n--- Accuracy per Fitzpatrick Scale (Disease) ---")

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
    scale_label = "Unknown/Missing (-1)" if scale == -1 else f"Scale {int(scale)}" # Handle -1 case from dataset
    if np.isnan(scale):
         scale_label = "NaN"
         indices = np.isnan(all_fitzpatrick_scales)
    else:
         indices = (all_fitzpatrick_scales == scale)

    if np.sum(indices) == 0:
        logger.info(f"{scale_label}: 0 samples.")
        continue # Skip if no samples for this scale

    subset_labels = all_disease_labels[indices]
    subset_preds = all_disease_preds[indices]

    scale_accuracy = accuracy_score(subset_labels, subset_preds) * 100
    num_samples = len(subset_labels)

    logger.info(f"{scale_label}: Accuracy = {scale_accuracy:.2f}% ({num_samples} samples)")
    fitzpatrick_results[scale_label] = {'accuracy': scale_accuracy, 'count': num_samples}

# You can also save these results if needed
results_df = pd.DataFrame.from_dict(fitzpatrick_results, orient='index')
results_df.to_csv("evaluation_fitzpatrick_accuracy_adversarial.csv")
logger.info("Fitzpatrick scale accuracy saved to evaluation_fitzpatrick_accuracy_adversarial.csv")

logger.info("\nEvaluation complete.")
# evaluate_adversarial_fitz.py
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# --- Define the Model Architecture (Copied from train_adversarial_fitz.py) ---
class GradientReversalLayer(torch.autograd.Function):
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
    def __init__(self, num_main_classes, num_fitzpatrick_classes, mlp_hidden_dim, grl_lambda=1.0):
        super().__init__()
        self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        encoder_output_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.main_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_main_classes)
        self.grl = GradientReversalLayer.apply
        self.grl_lambda = grl_lambda
        self.fitzpatrick_classifier = MLPHead(encoder_output_features, mlp_hidden_dim, num_fitzpatrick_classes)

    def forward(self, x):
        features = self.encoder(x)
        main_logits = self.main_classifier(features)
        reversed_features = self.grl(features, self.grl_lambda)
        fitzpatrick_logits = self.fitzpatrick_classifier(reversed_features)
        return main_logits, fitzpatrick_logits

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/InvDiff/Ankur_InvClass_Fitz/data/finalfitz17k/' # <<< CHANGE THIS
EVAL_METADATA_CSV = "/home/dse316/back/new/data/csv/fitzpatrick17k.csv"  # Use the same CSV for now, adjust if needed # <<< CHANGE THIS
LOG_FILE = 'Fitzpatrick17k/Final/evaluation_adversarial_effb3_fitzpatrick17k.log'
MODEL_SAVE_DIR = 'Fitzpatrick17k/Final/saved_models_adversarial_fitz' # Point to the adversarial model save directory
BEST_MODEL_NAME = '/home/dse316/back/InvDiff/FairAlign_Final_Results/mdkaif/Fitzpatrick17k/Final/saved_models_adversarial_fitz/adversarial_effb3_fitzpatrick17k_best.pth' # Point to the best adversarial model
BATCH_SIZE = 64 # Can often be larger for evaluation
NUM_MAIN_CLASSES = 3 # Based on 'three_partition_label'
NUM_FITZPATRICK_CLASSES = 6 # 1-6 mapped to 0-5
MLP_HIDDEN_DIM = 512 # Should match the training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
VAL_SIZE = 0.2  # Use the same split as training for consistent evaluation

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
logger.info("--- Evaluation Configuration ---")
logger.info(f"Data Root Directory: {DATA_ROOT_DIR}")
logger.info(f"Evaluation Metadata: {EVAL_METADATA_CSV}")
logger.info(f"Log File: {LOG_FILE}")
logger.info(f"Model Save Directory: {MODEL_SAVE_DIR}")
logger.info(f"Best Model Name: {BEST_MODEL_NAME}")
logger.info(f"Batch Size: {BATCH_SIZE}")
logger.info(f"Number of Main Classes: {NUM_MAIN_CLASSES}")
logger.info(f"Number of Fitzpatrick Classes: {NUM_FITZPATRICK_CLASSES}")
logger.info(f"MLP Hidden Dimension: {MLP_HIDDEN_DIM}")
logger.info("--- Starting Evaluation ---")

# --- Data Loading ---
logger.info("Loading evaluation dataset...")

full_dataset = pd.read_csv(EVAL_METADATA_CSV)

# Split into train and evaluation (using the same split as validation in training for now)
_, eval_df = train_test_split(
    full_dataset,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=full_dataset['three_partition_label']
)

# Define the same transformations as the validation set used during adversarial training
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FitzpatrickDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.data = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = {'non-neoplastic': 0, 'benign': 1, 'malignant': 2}
        logging.info(f"Label Mapping: {self.label_mapping}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['md5hash'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label_str = self.data.iloc[idx]['three_partition_label']
        if label_str in self.label_mapping:
            label = self.label_mapping[label_str]
        else:
            raise ValueError(f"Label '{label_str}' not found in the label mapping.")
        bias_label = int(self.data.iloc[idx]['fitzpatrick']) - 1 # Adjust to 0-indexed
        if not (0 <= bias_label <= 5):
            bias_label = -1 # Handle missing/invalid Fitzpatrick values
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        bias_label = torch.tensor(bias_label, dtype=torch.long)
        return image, label, bias_label

eval_dataset = FitzpatrickDataset(
    df=eval_df,
    root_dir=DATA_ROOT_DIR,
    transform=eval_transforms
)

eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

# --- Model Definition ---
logger.info("Initializing AdversarialDebiasingModel...")
model = AdversarialDebiasingModel(
    num_main_classes=NUM_MAIN_CLASSES,
    num_fitzpatrick_classes=NUM_FITZPATRICK_CLASSES,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    grl_lambda=1.0 # GRL lambda is used during training
).to(DEVICE)

# --- Load Trained Model ---
best_model_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    logger.info(f"Loaded best adversarial model from: {best_model_path}")
else:
    logger.error(f"Best adversarial model not found at: {best_model_path}. Evaluation cannot proceed.")
    exit()

# --- Evaluation Loop ---
model.eval()
total_correct_main = 0
total_samples = 0
all_preds_main = []
all_labels_main = []
all_fitzpatrick = []
running_loss_main = 0.0
criterion_main = nn.CrossEntropyLoss()

logger.info("Starting evaluation...")
with torch.no_grad():
    eval_progress = tqdm(eval_loader, desc="Evaluating", unit="batch")
    for images, labels_main, fitzpatrick in eval_progress:
        images = images.to(DEVICE)
        labels_main = labels_main.to(DEVICE)
        fitzpatrick = fitzpatrick.to(DEVICE)

        main_logits, _ = model(images) # Get both main and Fitzpatrick logits, but we primarily evaluate main task
        loss_main = criterion_main(main_logits, labels_main)
        running_loss_main += loss_main.item() * images.size(0)

        _, predicted_main = torch.max(main_logits, 1)
        total_samples += labels_main.size(0)
        total_correct_main += (predicted_main == labels_main).sum().item()

        all_preds_main.extend(predicted_main.cpu().numpy())
        all_labels_main.extend(labels_main.cpu().numpy())
        all_fitzpatrick.extend(fitzpatrick.cpu().numpy())

        eval_progress.set_postfix(loss=(running_loss_main / total_samples), acc=(100. * total_correct_main / total_samples))

avg_loss_main = running_loss_main / total_samples
accuracy_main = total_correct_main / total_samples * 100.0
logger.info(f"Overall Evaluation Loss (Main Task): {avg_loss_main:.4f}")
logger.info(f"Overall Evaluation Accuracy (Main Task): {accuracy_main:.2f}%")

# --- Classification Report and Confusion Matrix (Overall - Main Task) ---
class_names = ['non-neoplastic', 'benign', 'malignant']
report_main = classification_report(all_labels_main, all_preds_main, target_names=class_names, zero_division=0)
cm_main = confusion_matrix(all_labels_main, all_preds_main)

logger.info("\n--- Overall Classification Report (Main Task) ---")
logger.info(report_main)

logger.info("\n--- Overall Confusion Matrix (Main Task) ---")
logger.info(cm_main)

# --- Performance per Fitzpatrick Skin Type (Main Task) ---
logger.info("\n--- Performance per Fitzpatrick Skin Type (Main Task) ---")
for fitzpatrick_type in range(7):  # Assuming Fitzpatrick types are 0 to 5 (after 0-indexing), and -1 for missing
    if fitzpatrick_type == 6: # Original was 1-6
        continue # Adjust range as needed based on your dataset

    indices = np.where(np.array(all_fitzpatrick) == fitzpatrick_type)[0]
    if len(indices) > 0:
        fitz_labels_main = np.array(all_labels_main)[indices]
        fitz_preds_main = np.array(all_preds_main)[indices]

        fitz_accuracy_main = accuracy_score(fitz_labels_main, fitz_preds_main)
        fitz_report_main = classification_report(fitz_labels_main, fitz_preds_main, target_names=class_names, zero_division=0)
        fitz_cm_main = confusion_matrix(fitz_labels_main, fitz_preds_main)

        logger.info(f"\n--- Fitzpatrick Type: {fitzpatrick_type if fitzpatrick_type != -1 else 'Unknown/Missing'} ---")
        logger.info(f"Accuracy: {fitz_accuracy_main * 100:.2f}%")
        # logger.info("Classification Report:")
        # logger.info(fitz_report_main)
        # logger.info("Confusion Matrix:")
        # logger.info(fitz_cm_main)
    elif np.any(np.array(all_fitzpatrick) == fitzpatrick_type):
        logger.info(f"\n--- Fitzpatrick Type: {fitzpatrick_type if fitzpatrick_type != -1 else 'Unknown/Missing'} ---")
        logger.info("No samples found for this Fitzpatrick type in the evaluation set.")

logger.info("--- Evaluation Finished ---")
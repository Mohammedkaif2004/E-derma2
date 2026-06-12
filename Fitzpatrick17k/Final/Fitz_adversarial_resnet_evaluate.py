# FITZPATRICK17K_evaluate_adversarial_resnet.py
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights # Changed import
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

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

class AdversarialDebiasingModel(nn.Module):
    def __init__(self, num_disease_classes, num_fitzpatrick_classes, mlp_hidden_dim, grl_lambda=1.0):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder_output_features = self.encoder.fc.in_features # Changed to ResNet18's fc layer
        self.encoder.fc = nn.Identity() # Changed to ResNet18's fc layer
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
EVAL_METADATA_CSV = "/home/dse316/back/new/data/csv/fitzpatrick17k.csv"  # Use the same CSV for now, adjust if needed
LOG_FILE = 'Fitzpatrick17k/Final/evaluation_adversarial_resnet18_fitzpatrick17k.log'  # Renamed log file
MODEL_SAVE_DIR = 'Fitzpatrick17k/Final/saved_models_adversarial_fitz_resnet' # Updated model save directory
BEST_MODEL_NAME = '/home/dse316/back/InvDiff/FairAlign_Final_Results/mdkaif/Fitzpatrick17k/Final/saved_models_adversarial_fitz_resnet/adversarial_resnet18_fitzpatrick17k_best.pth'  # Renamed best model name
BATCH_SIZE = 512 # Using the batch size from the adversarial training
NUM_DISEASE_CLASSES = 3 # Updated to disease classes
NUM_FITZPATRICK_CLASSES = 6 # As defined in adversarial training
MLP_HIDDEN_DIM = 512 # As defined in adversarial training
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
logger.info(f"Number of Disease Classes: {NUM_DISEASE_CLASSES}")
logger.info(f"Number of Fitzpatrick Classes: {NUM_FITZPATRICK_CLASSES}")
logger.info(f"MLP Hidden Dimension: {MLP_HIDDEN_DIM}")
logger.info("--- Starting Evaluation ---")

# --- Data Loading ---
logger.info("Loading evaluation dataset...")

full_dataset = pd.read_csv(EVAL_METADATA_CSV)

# Split into train and evaluation (using the same split as validation in training)
_, eval_df = train_test_split(
    full_dataset,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=full_dataset['three_partition_label']
)

# Define the same transformations as the validation set in adversarial training
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
        self.fitzpatrick_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5} # Map Fitzpatrick 1-6 to 0-5
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
            bias = int(fitzpatrick_raw) - 1
            if not (0 <= bias <= 5):
                bias = -1
        except (ValueError, TypeError):
            bias = -1
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        bias = torch.tensor(bias, dtype=torch.long)
        return image, label, bias

eval_dataset = FitzpatrickDataset(
    df=eval_df,
    root_dir=DATA_ROOT_DIR,
    transform=eval_transforms
)

eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

# --- Model Definition ---
logger.info("Initializing Adversarial Debiasing Model...")
model = AdversarialDebiasingModel(
    num_disease_classes=NUM_DISEASE_CLASSES,
    num_fitzpatrick_classes=NUM_FITZPATRICK_CLASSES,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    grl_lambda=1.0 # Lambda doesn't matter for evaluation
).to(DEVICE)

# --- Load Trained Model ---
best_model_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    logger.info(f"Loaded best model from: {best_model_path}")
else:
    logger.error(f"Best model not found at: {best_model_path}. Evaluation cannot proceed.")
    exit()

# --- Evaluation Loop ---
model.eval()
total_correct = 0
total_samples = 0
all_preds = []
all_labels = []
all_fitzpatrick = []
running_loss = 0.0
criterion = nn.CrossEntropyLoss()

logger.info("Starting evaluation...")
with torch.no_grad():
    eval_progress = tqdm(eval_loader, desc="Evaluating", unit="batch")
    for images, labels, fitzpatrick in eval_progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        fitzpatrick = fitzpatrick.to(DEVICE)

        disease_logits, _ = model(images) # Get both disease and Fitzpatrick logits, but we evaluate on disease
        outputs = disease_logits # Use disease logits for evaluation
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_fitzpatrick.extend(fitzpatrick.cpu().numpy())

        eval_progress.set_postfix(loss=(running_loss / total_samples), acc=(100. * total_correct / total_samples))

avg_loss = running_loss / total_samples
accuracy = total_correct / total_samples * 100.0
logger.info(f"Overall Evaluation Loss: {avg_loss:.4f}")
logger.info(f"Overall Evaluation Accuracy (Disease): {accuracy:.2f}%")

# --- Classification Report and Confusion Matrix (Overall for Disease) ---
class_names = ['non-neoplastic', 'benign', 'malignant']
report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

logger.info("\n--- Overall Classification Report (Disease) ---")
logger.info(report)

logger.info("\n--- Overall Confusion Matrix (Disease) ---")
logger.info(cm)

# --- Performance per Fitzpatrick Skin Type (for Disease Classification) ---
logger.info("\n--- Performance per Fitzpatrick Skin Type (Disease Classification) ---")
for fitzpatrick_type in range(0, 6):  # Assuming Fitzpatrick types are 0 to 5 now
    indices = np.where(np.array(all_fitzpatrick) == fitzpatrick_type)[0]
    if len(indices) > 0:
        fitz_labels = np.array(all_labels)[indices]
        fitz_preds = np.array(all_preds)[indices]

        fitz_accuracy = accuracy_score(fitz_labels, fitz_preds)
        fitz_report = classification_report(fitz_labels, fitz_preds, target_names=class_names, zero_division=0)
        fitz_cm = confusion_matrix(fitz_labels, fitz_preds)

        logger.info(f"\n--- Fitzpatrick Type: {fitzpatrick_type} ---")
        logger.info(f"Accuracy: {fitz_accuracy * 100:.2f}%")
        # logger.info("Classification Report:")
        # logger.info(fitz_report)
        # logger.info("Confusion Matrix:")
        # logger.info(fitz_cm)
    else:
        logger.info(f"\n--- Fitzpatrick Type: {fitzpatrick_type} ---")
        logger.info("No samples found for this Fitzpatrick type in the evaluation set.")

logger.info("--- Evaluation Finished ---")
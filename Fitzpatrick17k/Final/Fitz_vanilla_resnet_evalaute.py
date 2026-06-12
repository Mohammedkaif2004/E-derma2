import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18  # Changed import
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration ---
DATA_ROOT_DIR = '/home/dse316/back/InvDiff/Ankur_InvClass_Fitz/data/finalfitz17k/'
EVAL_METADATA_CSV = "/home/dse316/back/new/data/csv/fitzpatrick17k.csv"  # Use the same CSV for now, adjust if needed
LOG_FILE = 'Fitzpatrick17k/Final/evaluation_resnet18_fitzpatrick17k.log'  # Renamed log file
MODEL_SAVE_DIR = 'Fitzpatrick17k/Final/saved_models'
BEST_MODEL_NAME = 'resnet18_fitzpatrick17k_best.pth'  # Renamed best model name
BATCH_SIZE = 1024
NUM_CLASSES = 3
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
logger.info(f"Number of Classes: {NUM_CLASSES}")
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

# Define the same transformations as the validation set
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
        bias = int(self.data.iloc[idx]['fitzpatrick'])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label, bias

eval_dataset = FitzpatrickDataset(
    df=eval_df,
    root_dir=DATA_ROOT_DIR,
    transform=eval_transforms
)

eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

# --- Model Definition ---
logger.info("Initializing ResNet18 model...")
model = resnet18(num_classes=NUM_CLASSES)  # Changed to resnet18
model.to(DEVICE)

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

        outputs = model(images)
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
logger.info(f"Overall Evaluation Accuracy: {accuracy:.2f}%")

# --- Classification Report and Confusion Matrix (Overall) ---
class_names = ['non-neoplastic', 'benign', 'malignant']
report = classification_report(all_labels, all_preds, target_names=class_names)
cm = confusion_matrix(all_labels, all_preds)

logger.info("\n--- Overall Classification Report ---")
logger.info(report)

logger.info("\n--- Overall Confusion Matrix ---")
logger.info(cm)

# --- Performance per Fitzpatrick Skin Type ---
logger.info("\n--- Performance per Fitzpatrick Skin Type ---")
for fitzpatrick_type in range(1, 7):  # Assuming Fitzpatrick types are 1 to 6
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
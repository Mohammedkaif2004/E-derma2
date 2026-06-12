import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class ISICDataset(Dataset):
    """
    ISIC Dataset Loader.

    Args:
        root_dir (str): Root directory of the ISIC dataset.
        split (str): 'train', 'val', or 'test'. Determines which metadata and image folder to use.
        transform (callable, optional): Optional transform to be applied on a sample.
        metadata_csv (str, optional): Path to the metadata CSV file. If None, uses default paths based on split.
        handle_unknown_labels (str): How to handle samples with unknown disease labels ('UNK').
            Options:
                - 'exclude':  Exclude samples with 'UNK' labels.
                - 'keep':     Keep samples with 'UNK' labels, mapping them to -1.
    """
    def __init__(self, root_dir, split='train', transform=None, metadata_csv=None, handle_unknown_labels='exclude'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.handle_unknown_labels = handle_unknown_labels

        # Determine metadata file and image directory
        if metadata_csv:
            metadata_file = metadata_csv
            if 'Train' in os.path.basename(metadata_file):
                self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input')
            elif 'Test' in os.path.basename(metadata_file) or 'Val' in os.path.basename(metadata_file):
                self.img_dir = os.path.join(root_dir, 'ISIC_2019_Test_Input')
            else:
                self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input' if split == 'train' else 'ISIC_2019_Test_Input')
                logger.warning(f"Could not infer image directory from metadata CSV name '{os.path.basename(metadata_file)}'. Using default for split '{split}': {self.img_dir}")
        else:
            if split == 'train':
                metadata_file = os.path.join(root_dir, 'ISIC_2019_Train_Metadata_with_Disease.csv')
                self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input')
            else:
                metadata_file = os.path.join(root_dir, 'ISIC_2019_Test_Metadata_with_Disease.csv')
                self.img_dir = os.path.join(root_dir, 'ISIC_2019_Test_Input')

        try:
            self.metadata_df = pd.read_csv(metadata_file)
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {metadata_file}")
            raise

        # --- Disease Mapping ---
        self.disease_mapping = {
            'MEL': 0, 'NV': 1, 'BCC': 2, 'BKL': 3, 'SCC': 4, 'DF': 5, 'VASC': 6, 'AK': 7
        }
        self.num_disease_classes = len(self.disease_mapping)

        # --- Fitzpatrick Scale Mapping ---
        self.fitzpatrick_mapping = {
            1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4, 6.0: 5
        }
        self.num_fitzpatrick_classes = 6

        # --- Preprocessing and Filtering ---
        required_cols = ['image', 'Disease', 'fitzpatrick_scale']
        if not all(col in self.metadata_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.metadata_df.columns]
            raise ValueError(f"Metadata CSV missing required columns: {missing}")

        # Handle missing Fitzpatrick values
        initial_fitz_count = len(self.metadata_df)
        self.metadata_df['fitzpatrick_scale'] = self.metadata_df['fitzpatrick_scale'].fillna(-1)
        if self.split == 'train':  # Only log/filter on training set to avoid surprises in eval
            if self.handle_unknown_labels == 'exclude':
                initial_unk_count = len(self.metadata_df)
                self.metadata_df = self.metadata_df[self.metadata_df['Disease'] != 'UNK'].copy()
                filtered_unk_count = len(self.metadata_df)
                logger.info(f"Train: Filtered out {initial_unk_count - filtered_unk_count} samples with 'UNK' disease labels.")
            else:
                self.metadata_df['Disease'] = self.metadata_df['Disease'].apply(lambda x: x if x != 'UNK' else -1)
                logger.info("Train: Mapped 'UNK' disease labels to -1.")

        filtered_fitz_count = len(self.metadata_df)
        if initial_fitz_count != filtered_fitz_count:
            logger.info(f"Modified Fitzpatrick scale handling changed dataset size by {initial_fitz_count - filtered_fitz_count}")

        # Ensure 'image' column exists
        if 'image' not in self.metadata_df.columns:
            raise ValueError("Metadata CSV must contain an 'image' column.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata_df.iloc[idx]
        img_name = row['image']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        # Get disease label
        disease = row['Disease']
        disease_label = self.disease_mapping.get(disease, -1)

        # Get Fitzpatrick label
        fitzpatrick_val = row['fitzpatrick_scale']
        fitzpatrick_label = self.fitzpatrick_mapping.get(float(fitzpatrick_val), -1)

        return image, torch.tensor(disease_label, dtype=torch.long), torch.tensor(fitzpatrick_label, dtype=torch.long)

    def get_disease_class_weights(self):
        """Calculates class weights for handling imbalance in disease classification."""

        # Handle potential -1 labels when calculating weights
        disease_labels = self.metadata_df['Disease']
        if (disease_labels == -1).any():
            disease_labels = disease_labels[disease_labels != -1]  # Exclude -1 for weight calc
        class_counts = disease_labels.value_counts().sort_index()

        # Ensure all classes in mapping are present, even if count is 0
        class_counts = class_counts.reindex(self.disease_mapping.keys(), fill_value=0)
        total_samples = len(self.metadata_df)
        weights = total_samples / (len(class_counts) * class_counts)
        weights[weights == float('inf')] = 0
        weights = weights / weights.mean()
        weights_tensor = torch.zeros(self.num_disease_classes)
        for disease, weight in weights.items():
            idx = self.disease_mapping.get(disease)
            if idx is not None:
                weights_tensor[idx] = weight
        return weights_tensor

    def get_fitzpatrick_class_weights(self):
        """Calculates class weights for handling imbalance in Fitzpatrick classification."""
        # Only consider non-missing values for weight calculation
        valid_fitz = self.metadata_df[self.metadata_df['fitzpatrick_scale'] != -1]['fitzpatrick_scale']
        class_counts = valid_fitz.value_counts().sort_index()
        # Ensure all classes 1-6 are present
        class_counts = class_counts.reindex(self.fitzpatrick_mapping.keys(), fill_value=0)
        total_samples = len(valid_fitz)
        if total_samples == 0:
            return torch.ones(self.num_fitzpatrick_classes)

        weights = total_samples / (len(class_counts) * class_counts)
        weights[weights == float('inf')] = 0
        weights = weights / weights.mean()
        weights_tensor = torch.zeros(self.num_fitzpatrick_classes)
        for f_val, weight in weights.items():
            idx = self.fitzpatrick_mapping.get(f_val)
            if idx is not None:
                weights_tensor[idx] = weight
        return weights_tensor
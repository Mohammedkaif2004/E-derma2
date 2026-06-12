# src/data/isic2.py
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
        filter_missing_fitzpatrick (bool): If True, samples with missing Fitzpatrick scale are removed.
                                          If False, missing values are replaced with -1.
    """
    def __init__(self, root_dir, split='train', transform=None, metadata_csv=None, filter_missing_fitzpatrick=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Standard size for many models
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])

        # Determine metadata file and image directory based on split
        if metadata_csv:
             metadata_file = metadata_csv
             # Try to infer image directory based on common naming conventions if not specified
             if 'Train' in os.path.basename(metadata_file):
                 self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input')
                 self.split = 'train' # Ensure split matches inferred usage
             elif 'Test' in os.path.basename(metadata_file) or 'Val' in os.path.basename(metadata_file):
                 self.img_dir = os.path.join(root_dir, 'ISIC_2019_Test_Input') # Assuming Test/Val use same input dir
                 self.split = 'val' # Ensure split matches inferred usage
             else:
                 # Fallback if naming doesn't match expected patterns
                 self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input' if split == 'train' else 'ISIC_2019_Test_Input')
                 logger.warning(f"Could not infer image directory from metadata CSV name '{os.path.basename(metadata_file)}'. Using default for split '{split}': {self.img_dir}")
        else:
            # Use default paths if metadata_csv is not provided
            if split == 'train':
                metadata_file = os.path.join(root_dir, 'ISIC_2019_Train_Metadata_with_Disease.csv')
                self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input')
            else:  # 'val' or 'test'
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
        # Map scale 1-6 to indices 0-5. Use -1 for missing/NaN values.
        self.fitzpatrick_mapping = {
            1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4, 6.0: 5
        }
        self.num_fitzpatrick_classes = 6 # Classes 0 through 5

        # --- Preprocessing and Filtering ---
        # Check for necessary columns
        required_cols = ['image', 'Disease', 'fitzpatrick_scale']
        if not all(col in self.metadata_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.metadata_df.columns]
            raise ValueError(f"Metadata CSV missing required columns: {missing}")

        # Handle missing Fitzpatrick values
        initial_count = len(self.metadata_df)
        if filter_missing_fitzpatrick:
            self.metadata_df = self.metadata_df.dropna(subset=['fitzpatrick_scale'])
            filtered_count = len(self.metadata_df)
            logger.info(f"Filtered out {initial_count - filtered_count} samples with missing Fitzpatrick scale.")
        else:
            # Replace NaN with -1, which will be used as ignore_index in loss
            self.metadata_df['fitzpatrick_scale'] = self.metadata_df['fitzpatrick_scale'].fillna(-1)
            logger.info("Replaced missing Fitzpatrick scale values with -1.")

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
            # Return dummy data or raise an error
            # Returning dummy data might hide issues during training
            # Let's return None and handle it in the collate_fn or dataloader loop if needed
            # Or raise an error:
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise # Re-raise the exception

        if self.transform:
            image = self.transform(image)

        # Get disease label
        disease = row['Disease']
        disease_label = self.disease_mapping.get(disease, -1) # Use -1 for unknown diseases? Or raise error?
        if disease_label == -1:
             logger.warning(f"Unknown disease type '{disease}' for image {img_name}. Mapping to -1.")
             # Depending on loss function, might need handling (e.g., ignore_index)
             # For now, let's assume all diseases in CSV are in mapping

        # Get Fitzpatrick label
        fitzpatrick_val = row['fitzpatrick_scale']
        if fitzpatrick_val == -1:
             fitzpatrick_label = -1 # Keep -1 for missing values if not filtered
        else:
             fitzpatrick_label = self.fitzpatrick_mapping.get(float(fitzpatrick_val), -1)
             if fitzpatrick_label == -1:
                 logger.warning(f"Unknown Fitzpatrick value '{fitzpatrick_val}' for image {img_name}. Mapping to -1.")

        return image, torch.tensor(disease_label, dtype=torch.long), torch.tensor(fitzpatrick_label, dtype=torch.long)

    def get_disease_class_weights(self):
        """Calculates class weights for handling imbalance in disease classification."""
        class_counts = self.metadata_df['Disease'].value_counts().sort_index()
        # Ensure all classes in mapping are present, even if count is 0
        class_counts = class_counts.reindex(self.disease_mapping.keys(), fill_value=0)
        total_samples = len(self.metadata_df)
        weights = total_samples / (len(class_counts) * class_counts)
        # Replace inf weights (for classes with 0 samples) with a large number or 0?
        # Let's normalize so weights sum roughly to num_classes
        weights[weights == float('inf')] = 0 # Or handle as needed
        weights = weights / weights.mean() # Normalize
        # Map weights to the numerical labels
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
             return torch.ones(self.num_fitzpatrick_classes) # Return equal weights if no valid samples

        weights = total_samples / (len(class_counts) * class_counts)
        weights[weights == float('inf')] = 0
        weights = weights / weights.mean() # Normalize
        # Map weights to the numerical labels (0-5)
        weights_tensor = torch.zeros(self.num_fitzpatrick_classes)
        for f_val, weight in weights.items():
            idx = self.fitzpatrick_mapping.get(f_val)
            if idx is not None:
                weights_tensor[idx] = weight
        return weights_tensor
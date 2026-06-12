# src/data/isic2.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ISICDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, metadata_csv=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # ISIC images may need larger size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])

        # Load metadata based on split
        if split == 'train':
            metadata_file = os.path.join(root_dir, 'ISIC_2019_Train_Metadata_with_Disease.csv')
            self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input')
        else:  # test or val
            metadata_file = os.path.join(root_dir, 'ISIC_2019_Test_Metadata_with_Disease.csv')
            self.img_dir = os.path.join(root_dir, 'ISIC_2019_Test_Input')

        self.metadata_df = pd.read_csv(metadata_file)

        # Map disease labels to numeric indices
        self.disease_mapping = {
            'MEL': 0,  # Melanoma
            'NV': 1,   # Nevus
            'BCC': 2,  # Basal Cell Carcinoma
            'BKL': 3,  # Benign Keratosis
            'SCC': 4,  # Squamous Cell Carcinoma
            'DF': 5,   # Dermatofibroma
            'VASC': 6, # Vascular Lesion
            'AK': 7    # Actinic Keratosis
        }

        # For binary classification (melanoma vs. non-melanoma)
        self.binary_mapping = {
            'MEL': 1,  # Melanoma (positive class)
            'NV': 0, 'BCC': 0, 'BKL': 0, 'SCC': 0, 'DF': 0, 'VASC': 0, 'AK': 0  # Non-melanoma (negative class)
        }

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        img_name = self.metadata_df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get disease label
        disease = self.metadata_df.iloc[idx]['Disease']

        # For multiclass classification
        label = self.disease_mapping.get(disease, 0)

        # For binary classification (melanoma vs. non-melanoma)
        binary_label = self.binary_mapping.get(disease, 0)

        # Get patient metadata (age, sex, anatomic site)
        age = self.metadata_df.iloc[idx].get('age_approx', -1) # Default to -1 if missing
        if pd.isna(age):
            age = -1

        sex_str = self.metadata_df.iloc[idx].get('sex', 'unknown')
        sex_encoding = -1
        if sex_str == 'male':
            sex_encoding = 1
        elif sex_str == 'female':
            sex_encoding = 0

        # Get Fitzpatrick scale
        fitzpatrick_scale = self.metadata_df.iloc[idx].get('fitzpatrick_scale', -1) # Default to -1 if missing
        if pd.isna(fitzpatrick_scale):
            fitzpatrick_scale = -1

        # Create metadata dictionary with numerical values
        metadata = {
            'age_approx': float(age),
            'sex': int(sex_encoding),
            'fitzpatrick_scale': float(fitzpatrick_scale)
        }

        return image, torch.tensor(label, dtype=torch.long), metadata # Changed to return multiclass label and metadata dictionary

    def get_label(self, img_name):
        # This method is not used in the current __getitem__, but kept for potential future use.
        image_id = os.path.splitext(img_name)[0]
        if image_id in self.metadata_df['image'].values:
            disease = self.metadata_df[self.metadata_df['image'] == image_id]['Disease'].iloc[0]
            return self.disease_mapping.get(disease, 0)
        return 0
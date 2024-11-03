from PIL import Image
import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import Dataset





class feature_dataset(Dataset):
    def __init__(self, df_feature, df_label):
        self.feature = df_feature
        self.label = df_label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):

        feature = self.feature[self.feature['path'] == self.label.iloc[idx, 0]].iloc[:, 3:]
        label_8_way = self.label['label_8_way'].iloc[idx]

        # Convert feature DataFrame to tensor
        feature = torch.tensor(feature.values, dtype=torch.float32).squeeze()
        return feature, label_8_way









class feature_extraction_dataset(Dataset):
    def __init__(self, df_file):
        self.file = pd.read_csv(df_file)
        

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        img_path = self.file.iloc[idx, 0]

        image = Image.open(img_path)
        
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        

        label_2_way = self.file.iloc[idx, -1]
        label_8_way = self.file.iloc[idx, -2]

        return image, img_path, label_8_way, label_2_way
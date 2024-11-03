import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import DeiTModel, DeiTConfig
import timm



class Feature_Extractor_Diet(nn.Module):
    def __init__(self):
        super(Feature_Extractor_Diet, self).__init__()
        

        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

    def forward(self, x):
        
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        
        return x2
    



class Feature_FC_layer_for_Diet(nn.Module):
    def __init__(self):
        super(Feature_FC_layer_for_Diet, self).__init__()

        self.Diet_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, 8)
            )
        
    
    def forward(self, x):
        x1 = self.Diet_fc(x)
        return x1
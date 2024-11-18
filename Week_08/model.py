import torch
import torch.nn as nn

class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # First layer reduces dimensionality
            nn.ReLU(),            # Activation function
            nn.Dropout(0.3),      # Regularization
            nn.Linear(256, 128),  # Second layer
            nn.ReLU(),            # Activation function
            nn.Linear(128, 90)    # Final layer for 90 classes
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        return self.classifier(features.float())
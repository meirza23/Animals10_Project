import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class AnimalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AnimalCNN, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze early layers (optional, but good for faster convergence on small datasets)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

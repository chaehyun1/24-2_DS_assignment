import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.dropout1(x)          
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.dropout2(x)                   

        x = x.view(x.size(0), -1)              
        x = F.relu(self.fc1(x))               
        x = self.fc2(x)                        
        
        return x


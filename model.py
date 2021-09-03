import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
    
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.net = models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.net(x)

class EfficientnetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.model.classifier = nn.Linear(in_features = self.model.classifier.in_features, out_features = num_classes, bias = True)
        
    def forward(self, x):
        return self.model(x)
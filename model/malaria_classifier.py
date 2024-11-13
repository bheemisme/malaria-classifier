import torchvision.models as models
import torch.nn as nn

class MalariaClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(MalariaClassifier, self).__init__()

        # Load pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the last fully connected layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x
    


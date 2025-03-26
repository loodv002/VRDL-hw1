import torch
import torch.nn as nn
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from datetime import datetime
from pathlib import Path

class Classifier(nn.Module):
    def __init__(self, n_classes: int):
        super(Classifier, self).__init__()

        self.resnet = resnext101_64x4d(
            weights=ResNeXt101_64X4D_Weights.DEFAULT
        )

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            n_classes
        )

        self.model_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    def forward(self, x):
        return self.resnet(x)
    
    @classmethod
    def load_checkpoint(cls, n_classes: int, checkpoint_path: str):
        model = cls(n_classes)
        model.load_state_dict(torch.load(checkpoint_path))
        model.model_name = Path(checkpoint_path).stem
        return model
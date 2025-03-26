import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from sklearn.utils.class_weight import compute_class_weight 
import yaml
import pickle
import sys

from classifier import Classifier, Trainer
from utils import check_n_parameters

if len(sys.argv) < 2:
    print('Config file not given, assume "./config.yml"')
    config_path = './config.yml'
else:
    config_path = sys.argv[2]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['path']['DATA_DIR']
MODEL_DIR = config['path']['MODEL_DIR']
OUTPUT_DIR = config['path']['OUTPUT_DIR']

TRAIN_DIR = f'{DATA_DIR}/train'
VAL_DIR = f'{DATA_DIR}/val'
TEST_DIR = f'{DATA_DIR}/test'

N_CLASSES = config['global']['N_CLASSES']

BATCH_SIZE = config['train']['BATCH_SIZE']
LEARNING_RATE = config['train']['LEARNING_RATE']
MAX_EPOCHES = config['train']['MAX_EPOCHES']
EARLY_STOP = config['train']['EARLY_STOP']

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
    ),
])

train_set = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=image_transform)
val_set = torchvision.datasets.ImageFolder(root=VAL_DIR, transform=image_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

class_weights = compute_class_weight('balanced', 
                                     classes=np.arange(N_CLASSES), 
                                     y=train_set.targets)
class_weights = torch.FloatTensor(class_weights)

assert train_set.class_to_idx == val_set.class_to_idx

model = Classifier(N_CLASSES)
print(f'Model name: {model.model_name}')
check_n_parameters(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

with open(f'{MODEL_DIR}/{model.model_name}_id_map.pkl', 'wb') as f:
	pickle.dump(train_set.class_to_idx, f)

trainer = Trainer()
trainer.train(
      model,
      train_loader,
      val_loader,
      MODEL_DIR,
      MAX_EPOCHES,
      criterion,
      optimizer,
      scheduler,
      EARLY_STOP,
)
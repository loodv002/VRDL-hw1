import torch
import pickle
from torch.utils.data import DataLoader

import argparse
import sys
import os
import yaml
from PIL import Image
from tqdm import tqdm
import zipfile

from classifier import Classifier, image_transform

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
parser.add_argument('--checkpoint', required=True, help='checkpoint name')
args = parser.parse_args()

config_path = args.config
checkpoint = args.checkpoint

print(f'Config file: {config_path}')
print(f'Checkpoint: {checkpoint}')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['path']['DATA_DIR']
MODEL_DIR = config['path']['MODEL_DIR']
OUTPUT_DIR = config['path']['OUTPUT_DIR']

N_CLASSES = config['global']['N_CLASSES']
BATCH_SIZE = config['train']['BATCH_SIZE']

TEST_DIR = f'{DATA_DIR}/test'

image_names = os.listdir(TEST_DIR)
images = []

print('Strat loading images')
for image_name in tqdm(image_names, ncols=100):
    image = Image.open(f'{TEST_DIR}/{image_name}').convert('RGB')
    image = image_transform(image)
    images.append(image)

test_loader = DataLoader(images, batch_size=BATCH_SIZE)
model = Classifier.load_checkpoint(N_CLASSES, 
                                   f'{MODEL_DIR}/{checkpoint}.pth')
device = torch.device('cuda' 
                      if torch.cuda.is_available()
                      else 'cpu')
model.to(device)
predict_ids = []

print('Start prediction')

model.eval()
with torch.no_grad():
    for images in tqdm(test_loader, ncols=100):
        images = images.to(device)

        output = model(images).float().cpu()
        pr_labels = torch.argmax(output, dim=1)

        predict_ids.extend(pr_labels.tolist())


idx_mapping_file_name = f'{model.model_name}_id_map.pkl'
with open(f'{MODEL_DIR}/{idx_mapping_file_name}', 'rb') as f:
    class_to_idx = pickle.load(f)

idx_to_class = {
    id: cls
    for cls, id in class_to_idx.items()
}

with open(f'{OUTPUT_DIR}/{checkpoint}.csv', 'w') as f:
    f.write('image_name,pred_label\n')
    for image_name, predict_id in zip(image_names, predict_ids):
        image_name_no_ext = image_name.split('.')[0]
        class_name = idx_to_class[predict_id]
        
        f.write(f'{image_name_no_ext},{class_name}\n')
output_zip_path = f'{OUTPUT_DIR}/{checkpoint}.zip'
output_csv_path = f'{OUTPUT_DIR}/{checkpoint}.csv'

with zipfile.ZipFile(output_zip_path, mode='w') as f:
    f.write(output_csv_path, 'prediction.csv')
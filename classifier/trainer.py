import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any

from .model import Classifier

class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' 
                                   if torch.cuda.is_available()
                                   else 'cpu')
        
    def _train_epoch(self,
                     model: Classifier,
                     train_loader: DataLoader,
                     criterion: nn.Module,
                     optimizer: optim.Optimizer) -> Tuple[float, float]:
        
        model.train()
            
        train_loss = 0
        train_correct = 0
        n_train_data = 0
        for images, gt_labels in tqdm(train_loader, ncols=100):
            images = images.to(self.device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output.cpu(), gt_labels)

            loss.backward()
            optimizer.step()

            pr_labels = torch.argmax(output.cpu(), dim=1)
            correct = (pr_labels == gt_labels).sum().item()
            
            train_loss += loss.item()
            train_correct += correct
            n_train_data += gt_labels.shape[0]

        train_loss /= len(train_loader)
        train_correct /= n_train_data
        return train_loss, train_correct

    def _val_epoch(self,
                   model: Classifier,
                   val_loader: DataLoader,
                   criterion: nn.Module) -> Tuple[float, float]:
        
        model.eval()

        val_loss = 0
        val_correct = 0
        n_val_data = 0
        with torch.no_grad():
            for images, gt_labels in tqdm(val_loader, ncols=100):
                images = images.to(self.device)

                output = model(images).float().cpu()
                pr_labels = torch.argmax(output, dim=1)

                loss = criterion(output, gt_labels)
                correct = (pr_labels == gt_labels).sum().item()
                
                val_loss += loss.item()
                val_correct += correct
                n_val_data += gt_labels.shape[0]

        val_loss /= len(val_loader)
        val_correct /= n_val_data
        
        return val_loss, val_correct

    def train(self,
              model: Classifier,
              train_loader: DataLoader,
              val_loader: DataLoader,
              checkpoint_dir: str,
              max_epoches: int,
              criterion: nn.Module,
              optimizer: optim.Optimizer,
              scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
              early_stop: bool = True,
              ):
        
        print(f'Train model by {self.device}')

        model = model.to(self.device)

        min_val_loss = float('inf')
        val_loss_increase_count = 0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(max_epoches):
            train_loss, train_accuracy = self._train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )

            val_loss, val_accuracy = self._val_epoch(
                model,
                val_loader,
                criterion,
            )

            print(f'Epoch {epoch} train loss: {train_loss:.3f}')
            print(f'Epoch {epoch} train accuracy: {train_accuracy * 100:.3f}%')
            print(f'Epoch {epoch} val loss: {val_loss:.3f}')
            print(f'Epoch {epoch} val accuracy: {val_accuracy * 100:.3f}%')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            if scheduler: scheduler.step()

            model_path = f'{checkpoint_dir}/{model.model_name}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_path)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                val_loss_increase_count = 0
            else:
                val_loss_increase_count += 1

            if val_loss_increase_count >= 2 and early_stop:
                print('Loss increased, training stopped.')
                break

        print(f'Train losses: {train_losses}')
        print(f'Val losses: {val_losses}')
        print(f'Train accuracies: {train_accuracies}')
        print(f'Val accuracies: {val_accuracies}')
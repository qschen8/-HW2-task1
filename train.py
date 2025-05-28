import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os
from models import get_model
import copy
import json


def get_config():
    return  {
    "data_dir": "data/caltech-101",
    "log_dir": "best_param/log",
    "checkpoint_dir": "best_param/ckpt",
    "batch_size": 256,
    "lr_backbone": 0.0001,
    "lr_fc": 0.0003,
    "num_epochs": 50,
    "pretrained": True,
    "num_workers": 16,
    "scheduler_step": 10,
    "scheduler_gamma": 0.1
  }


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def get_dataloaders(config):
    train_transform, test_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'train'), 
        train_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'test'),
        test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    return train_loader, test_loader

def prepare_model(config, device):
    model = get_model(pretrained=config['pretrained'])
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    return model.to(device)

def create_optimizer(model, config):
    model = model.module if isinstance(model, nn.DataParallel) else model
    
    decay_params = []
    no_decay_params = []
    fc_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            fc_params.append(param)
        else:
            if 'bn' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_groups = [
        {'params': decay_params, 'lr': config['lr_backbone'], 'weight_decay': 0.05},
        {'params': no_decay_params, 'lr': config['lr_backbone'], 'weight_decay': 0.0},
        {'params': fc_params, 'lr': config['lr_fc'], 'weight_decay': 0.05}
    ]
    
    return optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.999),  
        eps=1e-8
    )

def create_scheduler(optimizer, config):
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler_step'],
        gamma=config['scheduler_gamma']
    )

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    
    with tqdm(loader, desc="Training", leave=False) as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.inference_mode(), tqdm(loader, desc="Testing", leave=False) as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix(acc=f"{correct/total:.2%}")
    
    return correct / total

def save_checkpoint(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def main(config=None):
    if config is None:
        config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    train_loader, test_loader = get_dataloaders(config)
    model = prepare_model(config, device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    best_acc = 0.0
    
    # 训练
    for epoch in tqdm(range(config['num_epochs']), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        # 日志
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, os.path.join(config['checkpoint_dir'], "best_model.pth"))
        
        lrs = [f"{group['lr']:.1e}" for group in scheduler.optimizer.param_groups]
        tqdm.write(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2%} | "
            f"LRs: {'/'.join(lrs)}"
        )
    
    writer.close()
    print(f"\nBest Accuracy: {best_acc:.2%}")
    return best_acc

if __name__ == "__main__":
    main()
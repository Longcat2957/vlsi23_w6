import argparse
from functools import partial

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import baseDataset, get_dataloader
from network import modelA, modelB

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./MNIST")
parser.add_argument("--model_type", type=str, default="A", choices=["A", "B"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
parser.add_argument("--num-workers", type=int, default=2)

def get_optimizer(name:str):
    if name == "SGD":
        return partial(SGD, weight_decay=0.005, momentum=0.9)
    elif name == "Adam":
        return partial(Adam, weight_decay=0.005)
    else:
        raise NotImplementedError(name)

def get_model(type:str):
    assert type in ["A", "B"]
    if type == "A":
        return modelA()
    else:
        return modelB()

if __name__ == "__main__":
    opt = parser.parse_args()
    # check device
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')
    
    # if mps(Apple Silicon)
    # DEVICE = torch.device("mps")
    
    # make datasets
    train_loader, val_loader = get_dataloader(opt.root, opt.batch_size, opt.num_workers)
    
    # define network
    net = get_model(opt.model_type).to(DEVICE)
    
    # define criterion, optimizer
    criterion = CrossEntropyLoss()
    optimizer_type = get_optimizer(opt.optimizer)
    optimizer = optimizer_type(net.parameters(), lr=opt.lr)


    for epoch in range(1, opt.epochs+1):  # 'epochs'를 'epoch'로 수정
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()

        # Training phase
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        net.eval()

        # Validation phase
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch [{epoch}/{opt.epochs}]")  # 'EPOCHS'를 'opt.epochs'로 수정
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
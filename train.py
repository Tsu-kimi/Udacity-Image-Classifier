import argparse
import torch
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models
from os.path import isdir

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for a neural network.")
    parser.add_argument('--arch', type=str, default="vgg16", help="Model architecture to use, default is VGG16.")
    parser.add_argument('--save_dir', type=str, default="./checkpoint.pth", help="Path to save the trained model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument('--hidden_units', type=int, default=120, help="Number of units in the hidden layer.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for training.")
    parser.add_argument('--gpu', type=str, default="gpu", help="Enable GPU mode.")
    return parser.parse_args()

def data_transforms(train_dir, valid_dir, test_dir):
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
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

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    return train_data, valid_data, test_data

def get_loaders(train_data, valid_data, test_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return train_loader, valid_loader, test_loader

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        print("GPU available, training on GPU.")
        return torch.device("cuda")
    else:
        print("GPU not available or not requested, using CPU.")
        return torch.device("cpu")

def build_model(arch="vgg16", hidden_units=120):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs):
    steps = 0
    print_every = 30
    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss, accuracy = validate_model(model, valid_loader, criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")
                
                running_loss = 0
                model.train()

def validate_model(model, loader, criterion, device):
    valid_loss = 0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forw
import argparse
import torch
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models
from os.path import isdir

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for a neural network.")
    parser.add_argument('--arch', type=str, default="vgg16", help="Model architecture to use, default is VGG16.")
    parser.add_argument('--save_dir', type=str, default="./checkpoint.pth", help="Path to save the trained model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument('--hidden_units', type=int, default=120, help="Number of units in the hidden layer.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for training.")
    parser.add_argument('--gpu', type=str, default="gpu", help="Enable GPU mode.")
    return parser.parse_args()

def data_transforms(train_dir, valid_dir, test_dir):
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
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

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    return train_data, valid_data, test_data

def get_loaders(train_data, valid_data, test_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return train_loader, valid_loader, test_loader

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        print("GPU available, training on GPU.")
        return torch.device("cuda")
    else:
        print("GPU not available or not requested, using CPU.")
        return torch.device("cpu")

def build_model(arch="vgg16", hidden_units=120):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs):
    steps = 0
    print_every = 30
    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss, accuracy = validate_model(model, valid_loader, criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")
                
                running_loss = 0
                model.train()

def validate_model(model, loader, criterion, device):
    valid_loss = 0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forw

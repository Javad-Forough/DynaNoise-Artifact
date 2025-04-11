import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

###############################################################################
# Existing loading functions (CIFAR-10,ImageNet-10, and SST-2)
###############################################################################

def load_cifar10(data_dir, batch_size=128, img_size=None):
    if img_size is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def load_imagenet_subset(data_dir, dataset_name, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    full_path = os.path.join(data_dir, dataset_name)
    dataset = ImageFolder(root=full_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

###############################################################################
# NEW: Load SST-2 for DistilBERT
###############################################################################
def load_sst2(data_dir, batch_size=32):
    """
    Loads the GLUE 'sst2' split via Hugging Face datasets.
    Note: Requires 'datasets' library installed.
    """
    from datasets import load_dataset
    from transformers import DistilBertTokenizerFast
    
    # Download/load the SST-2 dataset
    sst2 = load_dataset("glue", "sst2")  # train, validation, test splits
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Simple encode function
    def encode(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

    # Map/encode
    sst2 = sst2.map(encode, batched=True)
    sst2.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_data = sst2['train']
    test_data = sst2['validation']  # we can treat the official dev set as test

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


###############################################################################
# Simple wrapper for load_sst2, returning train or test
###############################################################################
def get_sst2_loader(data_dir='data', batch_size=32, train=True):
    train_loader, test_loader = load_sst2(data_dir, batch_size)
    return train_loader if train else test_loader


###############################################################################
# Master get_data_loaders function
###############################################################################
def get_data_loaders(dataset_name, data_dir='data', batch_size=128, img_size=None):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return load_cifar10(data_dir, batch_size, img_size)
    elif dataset_name in ['imagenet-10', 'imagenet-100']:
        return load_imagenet_subset(data_dir, dataset_name, batch_size)
    elif dataset_name == 'sst2':
        # Return the train/test loader for SST2
        return load_sst2(data_dir, batch_size)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

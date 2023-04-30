import os
import random
import torch
import torchvision
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import functional as F
from glob import glob
import numpy as np
from PIL import Image
import time
import logging
from collections import OrderedDict
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import os
import urllib.request
import tarfile
import shutil
import subprocess
import zipfile



import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import requests
import timm
import torch
from torch import nn

from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


transform_224 = [transforms.Resize([224, 224]), transforms.ToTensor()]
transform_224_imagenetc = [transforms.Resize([224, 224]), transforms.RandomHorizontalFlip()]
transform_224_test = [transforms.Resize([224, 224]), transforms.ToTensor()]

transform_32 = [transforms.Resize([32, 32]), transforms.ToTensor()]
transform_32_imagenetc = [transforms.Resize([32, 32]), transforms.RandomHorizontalFlip()]
transform_32_test = [transforms.Resize([32, 32]), transforms.ToTensor()]


try:
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
except:
    raise ValueError('Wrong CUDA Device!')



def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.label

def copy_file(src, dst):
    shutil.copy(src, dst)

def download_file(url, path):
    filename = os.path.join(path, url.split('/')[-1])
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
    else:
        os.makedirs(path, exist_ok=True)  # create directory if it doesn't exist
        urllib.request.urlretrieve(url, filename)
        print(f"File {filename} downloaded successfully.")


def getLoaders(in_dataset,out_dataset,batch_size):

    num_classes = {
        'CIFAR10': 10,
        'CIFAR100': 20,
        'MNIST': 10,
        'FashionMNIST': 10,
    }[in_dataset]

    if in_dataset in ["MNIST", "FashionMNIST"]:
        transform_32.insert(1, transforms.Grayscale(num_output_channels=3))
        transform_32_imagenetc.insert(1, transforms.Grayscale(num_output_channels=3))

    if out_dataset in ["MNIST", "FashionMNIST"]:
        transform_32_test.insert(1, transforms.Grayscale(num_output_channels=3))

    train_dataset = eval(f'torchvision.datasets.{in_dataset}("./{in_dataset}", train=True, download=True, transform=transforms.Compose(transform_32))')
    test_dataset_in = eval(f'torchvision.datasets.{in_dataset}("./{in_dataset}", train=False, download=True, transform=transforms.Compose(transform_32))')
    
    if out_dataset in ['MNIST','CIFAR10','CIFAR100']:
      test_dataset_out = eval(f'torchvision.datasets.{out_dataset}("./{out_dataset}", train=False, download=True, transform=transforms.Compose(transform_32_test))')
    
    elif out_dataset =='LSUN':
        

        os.chmod('./downloadLSUN.sh', 0o755)
        result = subprocess.run('./downloadLSUN.sh', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        test_dataset_out=eval(f"torchvision.datasets.ImageFolder(root = './LSUN_resize',transform=transforms.Compose(transform_32_test))")

    
    elif out_dataset =='Places365':
    
        URLs=['https://dl.dropboxusercontent.com/s/3pwqsyv33f6if3z/val_256.tar','https://dl.dropboxusercontent.com/s/gaf1ygpdnkhzyjo/places365_val.txt','https://dl.dropboxusercontent.com/s/enr71zpolzi1xzm/categories_places365.txt']
        for url in URLs:
            download_file(url,'./downloaded_files')
        
        
        tar = tarfile.open('./downloaded_files/val_256.tar', 'r')
        tar.extractall('./')
        tar.close()
        
        copy_file('./downloaded_files/places365_val.txt','./val_256/places365_val.txt')
        copy_file('./downloaded_files/categories_places365.txt','./val_256/categories_places365.txt')


        test_dataset_out = eval(f"torchvision.datasets.Places365('./val_256', split = 'val',small = True, transform=transforms.Compose(transform_32_test))")
        

    elif out_dataset =='COIL-100':
      # test_dataset_out = eval(f'torchvision.datasets.{out_dataset}("./{out_dataset}", train=False, download=True, transform=transforms.Compose(transform_224_test))')

        download_file('http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip','./downloaded_files')
        with zipfile.ZipFile('./downloaded_files/coil-100.zip', 'r') as zip_ref:
            zip_ref.extractall('./coil-100')

        test_dataset_out=eval(f"torchvision.datasets.ImageFolder(root = './coil-100', transform =transforms.Compose(transform_32_test))")


    print(f"test_dataset_out : {len(test_dataset_out)}")
    
    if in_dataset == 'CIFAR100':
        train_dataset.targets = sparse2coarse(train_dataset.targets)
        test_dataset_in.targets = sparse2coarse(test_dataset_in.targets)

    # Change out dataset targets
    test_dataset_out.targets = [num_classes for target in train_dataset.targets]

    # Load ImageNet dataset
    imagenet_data = torch.from_numpy(np.load("./OE_imagenet_32x32.npy"))

    # Get Number of Unique Labels
    unique_train_labels = set(train_dataset.targets if in_dataset not in ["MNIST", "FashionMNIST"] else [t.item() for t in train_dataset.targets])
    num_unique_train_labels = len(unique_train_labels)

    # Randomly select k images from the ImageNet dataset
    k = len(train_dataset) // num_unique_train_labels
    print(k, len(train_dataset), num_unique_train_labels)
    selected_indices = random.sample(range(len(imagenet_data)), k)
    selected_imagenet_data = imagenet_data[selected_indices]

    # Create a dataset with the selected ImageNet images and the last label (num_classes - 1)
    imagenet_label = num_classes
    imagenet_train_data = ImageNetDataset(selected_imagenet_data, imagenet_label, transforms.Compose(transform_32_imagenetc))

    trainset = torch.utils.data.ConcatDataset([train_dataset, imagenet_train_data])
    testset = torch.utils.data.ConcatDataset([test_dataset_in, test_dataset_out])

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size,num_workers=2)
    testloader = DataLoader(testset, shuffle=False, batch_size=batch_size//2,num_workers=2)
    shuffled_testloader = DataLoader(testset, shuffle=True, batch_size=batch_size//2,num_workers=2)


    print(f"Length of train dataset: {len(trainset)}")
    print(f"Length of test dataset: {len(testset)}")
    
    return trainloader,testloader

def visualize_samples(dataloader, label, n, title="Sample"):
    matching_samples = []
    non_matching_samples = []
    
    def to_3_channels(image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        return image
    
    # Collect n x n samples
    for images, labels in dataloader:
        for i, l in enumerate(labels):
            image = to_3_channels(images[i])
            if len(matching_samples) < n * n and l == label:
                matching_samples.append(image)
            elif len(non_matching_samples) < n * n and l != label:
                non_matching_samples.append(image)
            if len(matching_samples) == n * n and len(non_matching_samples) == n * n:
                break
        if len(matching_samples) == n * n and len(non_matching_samples) == n * n:
            break

    matching_grid = F.to_pil_image(torchvision.utils.make_grid(matching_samples, nrow=n))
    non_matching_grid = F.to_pil_image(torchvision.utils.make_grid(non_matching_samples, nrow=n))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    fig.patch.set_alpha(0)
    fig.suptitle(title, fontsize=16)

    axs[0].imshow(np.array(non_matching_grid))
    axs[0].set_title(f'In Distribution', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(np.array(matching_grid))
    axs[1].set_title(f'Out Distribution', fontsize=14)
    axs[1].axis('off')

    plt.show()
    
    
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

def auc_softmax_adversarial(model, test_loader, test_attack, epoch:int, device, num_classes):

    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

    with tqdm(test_loader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)

            adv_data = test_attack(data, target)
            output = model(adv_data)

            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()

            probs = soft(output).squeeze()
            anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()

            target = target == num_classes
            
            test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)
    accuracy = accuracy_score(test_labels, preds, normalize=True)

    if is_train:
        model.train()
    else:
        model.eval()

    return auc, accuracy

def auc_softmax(model, test_loader, epoch:int, device, num_classes):

    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for i, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                output = model(data)

                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                preds += predictions.detach().cpu().numpy().tolist()

                probs = soft(output).squeeze()
                anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()

                target = target == num_classes
                
                test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)
    accuracy = accuracy_score(test_labels, preds, normalize=True)

    if is_train:
        model.train()
    else:
        model.eval()

    return auc, accuracy

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

def auc_MSP(model, test_loader, epoch:int, device, num_classes):
    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []
    test_labels_acc = []

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for i, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                output = model(data)

                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                preds += predictions.detach().cpu().numpy().tolist()

                # probs = soft(output).squeeze()
                # anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()

                probs = soft(output) 
                max_probabilities,_ = torch.max(probs[:,:num_classes] , dim=1)
                anomaly_scores+=max_probabilities.detach().cpu().numpy().tolist()
                # anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()  


                test_labels_acc += target.detach().cpu().numpy().tolist()
                target = target == num_classes
                
                test_labels += target.detach().cpu().numpy().tolist()
    anomaly_scores=[x * -1 for x in anomaly_scores]
    auc = roc_auc_score(test_labels,  anomaly_scores)
    accuracy = accuracy_score(test_labels_acc, preds, normalize=True)

    if is_train:
        model.train()
    else:
        model.eval()

    return auc, accuracy
def auc_MSP_adversarial(model, test_loader, test_attack, epoch:int, device, num_classes):
    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []
    test_labels_acc = []
    with tqdm(test_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for i, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                adv_data = test_attack(data, target)
                output = model(adv_data)

                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                preds += predictions.detach().cpu().numpy().tolist()

                # probs = soft(output).squeeze()
                # anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()

                probs = soft(output) 
                max_probabilities,_ = torch.max(probs[:,:num_classes] , dim=1)
                anomaly_scores+=max_probabilities.detach().cpu().numpy().tolist()
                # anomaly_scores += probs[:, num_classes].detach().cpu().numpy().tolist()  


                test_labels_acc += target.detach().cpu().numpy().tolist()
                target = target == num_classes
                
                test_labels += target.detach().cpu().numpy().tolist()
    anomaly_scores=[x * -1 for x in anomaly_scores]
    auc = roc_auc_score(test_labels,  anomaly_scores)
    accuracy = accuracy_score(test_labels_acc, preds, normalize=True)

    if is_train:
        model.train()
    else:
        model.eval()

    return auc, accuracy
    
def lr_schedule(learning_rate:float, t:float, max_epochs:int):
    if t / max_epochs < 0.5:
        return learning_rate
    elif t / max_epochs < 0.75:
        return learning_rate / 10.
    elif t / max_epochs < 0.875:
        return learning_rate / 100.
    else:
        return learning_rate / 1000.
    
import os

def unique_filename(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename

import csv





def run(csv_filename, model, train_attack, test_attack, trainloader, testloader, test_step:int, max_epochs:int, device, loss_threshold, num_classes,optimizer,lr):

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    criterion = nn.CrossEntropyLoss()
    init_epoch = 0

    clean_aucs = []
    adv_aucs = []

    # Generate a unique filename if the file already exists
    csv_filename = unique_filename(csv_filename)
    
    print(f'Results Will be Saved To {csv_filename}.')

    # Write the header to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Epoch', 'AUC-Clean', 'Accuracy-Clean', 'AUC-Adversarial', 'Accuracy-Adversarial', 'Train-Loss', 'Train-Accuracy']
        csvwriter.writerow(header)

    
    print(f'Starting Run from epoch {init_epoch}')
    
    for epoch in range(init_epoch, max_epochs+1):

        torch.cuda.empty_cache()

        logs = {}


        
        if epoch < max_epochs:
            print(f'====== Starting Training on epoch {epoch}')
            train_accuracy, train_loss = train_one_epoch(epoch=epoch,\
                                                                    max_epochs=max_epochs, \
                                                                    model=model,\
                                                                    optimizer=optimizer,
                                                                    criterion=criterion,\
                                                                    trainloader=trainloader,\
                                                                    train_attack=train_attack,\
                                                                    lr=lr,\
                                                                    device=device)
            
            print("train accuracy is ", train_accuracy)
            print("train loss is ", train_loss)
            if train_loss < loss_threshold:
                break


        if epoch % test_step == 0 :

            test_auc = {}
            test_accuracy = {}

            print(f'AUC & Accuracy Vanila - Started...')
            clean_auc, clean_accuracy  = auc_softmax(model=model, epoch=epoch, test_loader=testloader, device=device, num_classes=num_classes)
            test_auc['Clean'], test_accuracy['Clean'] = clean_auc, clean_accuracy
            print(f'AUC Vanila - score on epoch {epoch} is: {clean_auc * 100}')
            print(f'Accuracy Vanila -  score on epoch {epoch} is: {clean_accuracy * 100}')
            logs[f'AUC-Clean'], logs[f'Accuracy-Clean'] = clean_auc, clean_accuracy

            attack_name = 'PGD-10'
            attack = test_attack
            print(f'AUC & Accuracy Adversarial - {attack_name} - Started...')
            adv_auc, adv_accuracy = auc_softmax_adversarial(model=model, epoch=epoch, test_loader=testloader, test_attack=attack, device=device, num_classes=num_classes)
            print(f'AUC Adversairal {attack_name} - score on epoch {epoch} is: {adv_auc * 100}')
            print(f'Accuracy Adversairal {attack_name} -  score on epoch {epoch} is: {adv_accuracy * 100}')

            torch.cuda.empty_cache()

            clean_aucs.append(clean_auc)
            adv_aucs.append(adv_auc)
            
            # Update the row with the test results
            with open(csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                row = [epoch, clean_auc, clean_accuracy, adv_auc, adv_accuracy, train_loss if epoch < max_epochs else '', train_accuracy if epoch < max_epochs else '']
                csvwriter.writerow(row)



    return clean_aucs, adv_aucs



def train_one_epoch(epoch, max_epochs, model, optimizer, criterion, trainloader, train_attack, lr, device):

    soft = torch.nn.Softmax(dim=1)

    preds = []
    true_labels = []
    running_loss = 0
    accuracy = 0

    model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}/{max_epochs}")
            updated_lr = lr_schedule(learning_rate=lr, t=epoch + (i + 1) / len(tepoch), max_epochs=max_epochs)
            optimizer.param_groups[0].update(lr=updated_lr)

            data, target = data.to(device), target.to(device)
            target = target.type(torch.LongTensor).cuda()

            # Adversarial attack on every batch
            data = train_attack(data, target)

            # Zero gradients for every batch
            optimizer.zero_grad()

        
            output = model(data)

            # Compute the loss and its gradients
            loss = criterion(output, target)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            true_labels += target.detach().cpu().numpy().tolist()

            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()
            correct = (torch.tensor(preds) == torch.tensor(true_labels)).sum().item()
            accuracy = correct / len(preds)

            probs = soft(output).squeeze()

            running_loss += loss.item() * data.size(0)

            tepoch.set_postfix(loss=running_loss / len(preds), accuracy=100. * accuracy)

    return  accuracy_score(true_labels, preds, normalize=True), \
            running_loss / len(preds)
    
    
    
import logging

class GeneralLogger:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.log_file = log_file
        
        # create a file handler and set its level to INFO
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)

        # add the handler to the logger
        self.logger.addHandler(handler)

    def log(self, message):
        # log the message
        self.logger.info(message)
    

def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """

    if  os.path.isfile(fname_save):
        print("File already exists.")
        return 
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))    
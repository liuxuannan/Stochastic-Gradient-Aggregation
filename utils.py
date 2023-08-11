'''
Functions for:
- Loading models, datasets
- Evaluating on datasets with or without UAP
'''

import multiprocessing
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision

from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
 
def model_imgnet(model_name):
    model = eval("torchvision.models.{}(pretrained=True)".format(model_name))
    model = nn.DataParallel(model).cuda()
    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)
    model = nn.Sequential(normalize, model)
    model = model.cuda()
    print("Model loading complete.")
    
    return model


# dataloader for ImageNet
def loader_imgnet(dir_data, nb_images = 50000, batch_size = 100, model_dimension = 256,center_crop=224):
    val_transform = transforms.Compose([
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
    ])

    val_dataset = ImageFolder(dir_data, val_transform)

    # Random subset if not using the full 50,000 validation set
    if nb_images < 50000:
        np.random.seed(0)
        sample_indices = np.random.permutation(range(50000))[:nb_images]
        val_dataset = Subset(val_dataset, sample_indices)
    
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,                              
        shuffle = False,
        num_workers = 0
    )
    
    return dataloader

# Evaluate model on data with or without UAP
# Assumes data range is bounded by [0, 1]
def evaluate(model, loader, uap = None, n = 5,batch_size =None, DEVICE=None):
    probs, labels, y_out= [], [], []
    model.eval()
    
    if uap is not None:
        batch_size = batch_size
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1]).to(DEVICE)
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data[0].to(DEVICE)
            y_val = data[1].to(DEVICE)
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val), dim = 1)
            else:
                y_ori = torch.nn.functional.softmax(model(x_val), dim = 1)
                perturbed = torch.clamp((x_val + uap), 0, 1) # clamp to [0, 1]
                out = torch.nn.functional.softmax(model(perturbed), dim = 1)

            probs.append(out.cpu().numpy())
            labels.append(y_val.cpu())
            y_out.append(y_ori.cpu().numpy())

    # Convert batches to single numpy arrays
    probs = np.array([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    y_out = np.array([s for l in y_out for s in l])

    # Extract top 5 predictions for each example
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float32)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis = 1)]

    y_top = np.argpartition(-y_out, n, axis=1)[:, :n]
    y_top_probs = y_out[np.arange(y_out.shape[0])[:, None], y_top].astype(np.float32)
    y_outputs = y_top[range(len(y_top)), np.argmax(y_top_probs, axis=1)]
        
    return top, top_probs, top1acc, top5acc, outputs, labels, y_outputs
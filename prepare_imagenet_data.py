import numpy as np
import os
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image


def preprocess_image(image_paths = None,model_dimension=256,center_crop=224):

    img = Image.open(image_paths).convert('RGB')
    train_transform = transforms.Compose([
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
    ])
    img = train_transform(img)

    return img

class subDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)

def create_imagenet_npy(path_train_imagenet, len_batch=10000,model_dimension=256,center_crop=224):

    # path_train_imagenet = '/../imagenet/train';

    sz_img = [center_crop, center_crop]
    num_channels = 3
    num_classes = 1000

    im_array = np.zeros([len_batch] + [num_channels]+sz_img, dtype=np.float32)
    num_imgs_per_batch = int(len_batch / num_classes)

    dirs = [x[0] for x in os.walk(path_train_imagenet)]
    dirs = dirs[1:]

    # Sort the directory in alphabetical order (same as synset_words.txt)
    dirs = sorted(dirs)

    it = 0
    Matrix = [0 for x in range(1000)]

    for d in dirs:
        for _, _, filename in os.walk(d):
            Matrix[it] = filename
        it = it+1

    it = 0
    # Load images, pre-process, and save
    for k in range(num_classes):
        for u in range(num_imgs_per_batch):
            print('Processing image number ', it)
            path_img = os.path.join(dirs[k], Matrix[k][u])
            image = preprocess_image(path_img,model_dimension,center_crop)
            im_array[it:(it+1), :, :, :] = image
            it = it + 1
    Imageset = im_array
    return Imageset
import os

import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch import nn, optim

from dataset import *
from engine import train_fn
from metrics import check_accuracy


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything(42)
    DATA_DIR = ""
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    MASKS_DIR = os.path.join(DATA_DIR, 'masks')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 192

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.ColorJitter(p=0.2),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ])
    train_loader, val_loader = get_loader(input_dir=IMAGES_DIR, mask_dir=MASKS_DIR,
                                          batch_size=BATCH_SIZE, train_transform=train_transform,
                                          val_transform=test_transform)
    model = smp.Unet(encoder_name='efficientnet-b3', in_channels=3, classes=1, activation=None).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print('#'*20, str(epoch))
        train_fn(train_loader, model, optimizer, loss_fn)
        check_accuracy(val_loader, model, DEVICE)

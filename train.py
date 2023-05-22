import os

import numpy as np
import pandas as pd
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch import nn, optim

from dataset import *
from engine import *
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
    loss_train, accuracy_train, precision_train, recall_train, dice_train, dice_train, iou_train = [], [], [], [], [], [], []
    loss_test, accuracy_test, precision_test, recall_test, dice_test, dice_test, iou_test = [], [], [], [], [], [], []
    for epoch in range(NUM_EPOCHS):
        print('#' * 20, str(epoch))
        score_train = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        score_test = evaluate_fn(val_loader, model, loss_fn, DEVICE)

        loss_train.append(score_train['loss'])
        accuracy_train.append(score_train['accuracy_score'])
        precision_train.append(score_train['precision_score'])
        recall_train.append(score_train['recall_score'])
        dice_train.append(score_train['dice_score'])
        iou_train.append(score_train['iou_score'])

        loss_test.append(score_test['loss'])
        accuracy_test.append(score_test['accuracy_score'])
        precision_test.append(score_test['precision_score'])
        recall_test.append(score_test['recall_score'])
        dice_test.append(score_test['dice_score'])
        iou_test.append(score_test['iou_score'])

    history = {
        'epoch': epoch,
        'loss_train': loss_train,
        'loss_test': loss_test,
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'iou_score_train': iou_train,
        'iou_score_test': iou_test,
        'dice_score_train': iou_train,
        'dice_score_test': dice_test,
    }

    experiment_name = 'unet_efficient_b3'
    SAVE_DIR = Path('/content/drive/MyDrive/DEEP LEARNING PROJECT/PRAYITNO/experiments/')
    torch.save(model, os.path.join(SAVE_DIR, f"model_{experiment_name}.pth"))
    pd.DataFrame.from_dict(history).to_csv(os.path.join(SAVE_DIR, f"history_{experiment_name}.csv"), index=False)


import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, output_dir, is_train, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        x = round(len(os.listdir(input_dir)) * 0.8)
        if is_train:
            self.images = os.listdir(input_dir)[:x]
        else:
            self.images = os.listdir(input_dir)[x:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.input_dir, self.images[item])
        mask_path = os.path.join(self.output_dir, self.images[item])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255

        if self.transform:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations['image']
            mask = augmentations['mask']
        return img, mask


def get_loader(input_dir, mask_dir, batch_size, train_transform, val_transform):
    train_ds = SegmentationDataset(input_dir=input_dir, output_dir=mask_dir,
                                   is_train=True, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = SegmentationDataset(input_dir=input_dir, output_dir=mask_dir,
                                 is_train=False, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

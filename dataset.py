import os

import numpy as np
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
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


class CarvanaCarSegmentation(Dataset):
    def __init__(self, images_dir, masks_dir, is_train, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        x = round(len(os.listdir(images_dir)) * .8)
        if is_train:
            self.images = os.listdir(images_dir)[:x]
        else:
            self.images = os.listdir(images_dir)[x:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.images_dir, self.images[item])
        mask_path = os.path.join(self.masks_dir, self.images[item][:-4] + '_mask.gif')
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask


def get_carvana_loader(input_dir, mask_dir, batch_size, train_transform, val_transform):
    train_ds = CarvanaCarSegmentation(images_dir=input_dir, masks_dir=mask_dir,
                                      is_train=True, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = CarvanaCarSegmentation(images_dir=input_dir, masks_dir=mask_dir,
                                    is_train=False, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class PlanktonSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_file='_annotations.coco.json', transform=None):
        super(PlanktonSegmentationDataset, self).__init__()
        self.data_dir = data_dir
        self.annotations = COCO(os.path.join(data_dir, annotation_file))
        self.image_ids = self.annotations.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_info = self.annotations.loadImgs(self.image_ids[item])[0]
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        annotation_ids = self.annotations.getAnnIds(imgIds=self.image_ids[item])
        annotations = self.annotations.loadAnns(annotation_ids)
        mask = Image.new('L', image.size)
        for ann in annotations:
            seg_mask = self.annotations.annToMask(ann)
            seg_mask = Image.fromarray(seg_mask.astype('uint8') * 255)
            mask.paste(seg_mask, (0, 0), mask=seg_mask)

        image = np.array(image, dtype=np.float32) / 255
        mask = np.array(mask, dtype=np.float32) / 255

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask


def get_plankton_loader(input_dir="", batch_size=16, is_train=True, transform=None):
    train_ds = PlanktonSegmentationDataset(images_dir=input_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=is_train)
    return train_loader


# if __name__ == '__main__':
#     LEARNING_RATE = 3e-4
#     BATCH_SIZE = 64
#     NUM_EPOCHS = 10
#     IMAGE_HEIGHT = 224
#     IMAGE_WIDTH = 224
#     DATA_DIR = Path('/content/drive/MyDrive/DEEP LEARNING PROJECT/PRAYITNO/datasets/plankton_cocov2')
#     TRAIN_DIR = os.path.join(DATA_DIR, 'train')
#     TEST_DIR = os.path.join(DATA_DIR, 'test')
#     VAL_DIR = os.path.join(DATA_DIR, 'valid')
#     train_loader = get_plankton_loader(input_dir=TRAIN_DIR, batch_size=BATCH_SIZE,is_train=True,
#                                        transform=train_transform)
#     test_loader = get_plankton_loader(input_dir=TRAIN_DIR, batch_size=BATCH_SIZE,is_train=True,
#                                        transform=test_transform)
#     val_loader = get_plankton_loader(input_dir=TRAIN_DIR, batch_size=BATCH_SIZE,is_train=True,
#                                        transform=test_transform)


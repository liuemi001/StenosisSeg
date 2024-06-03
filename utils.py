import timm
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torch.nn as nn
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms.functional import pil_to_tensor

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import cv2
from torch.utils.data import Dataset
import warnings


def load_yolo_annotations(file_path, img_width, img_height):
    """
    Load YOLO annotations from a file and convert them to bounding boxes and polygons.
    Returns: a list of (class_id, polygon_mask) tuples
    """
    boxes = []
    labels = []
    masks = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            # record class label
            label = int(parts[0])
            labels.append(label)

            # create pixelwise mask from YOLO polygon annotation
            polygon = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
            polygon[:, 0] *= img_width
            polygon[:, 1] *= img_height
            polygon_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            polygon = polygon.astype(np.int32)
            cv2.fillPoly(polygon_mask, [polygon], color=1)
            masks.append(polygon_mask)

            # create bounding box based on polygon coordinates
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]
            x_min = np.min(x_coords)
            x_max = np.max(x_coords)
            y_min = np.min(y_coords)
            y_max = np.max(y_coords)
            coordinates = [x_min, y_min, x_max, y_max]
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
            boxes.append(bbox)

    boxes = torch.stack(boxes, dim=0)
    labels = torch.tensor(labels, dtype=torch.int64)
    masks = torch.from_numpy(np.stack(masks, axis=0))
  
    return boxes, labels, masks


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def save_checkpoint(model, optimizer, epoch, iteration, loss, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # iteration = checkpoint['iteration']
    print(f"Checkpoint loaded from epoch {epoch} with loss {loss:.4f}")
    return model, optimizer, epoch, loss


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        img_dir = os.path.join(data_dir, 'train/images')
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return 997

    def __getitem__(self, idx):
        img_name = str(idx + 1) + '.png'
        ann_name = str(idx + 1) + '.txt'
        image_path = os.path.join(self.data_dir, 'train/images', img_name)  # Replace 'example.jpg' with your image file
        annotation_path = os.path.join(self.data_dir, 'train/labels', ann_name)  # Replace 'example.txt' with your annotation file

        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = pil_to_tensor(image)
        image = image.to(torch.float32)
        image = image / 255.0

        boxes, labels, masks = load_yolo_annotations(annotation_path, 224, 224)
        targets = {'image_id': torch.tensor([idx + 1]), 'boxes': boxes, 'masks': masks, 'labels': labels}

        return image, targets
import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    """
    A Dataset class to load images and their labels for object detection.
    """

    def __init__(self, image_dir, label_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        """
        # Load Image
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Read the image and convert it to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Labels
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        bboxes = []
        class_ids = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # YOLO format(class_id, x_center, y_center, width, height)
                class_id, x_center, y_center, w, h = [float(p) for p in parts]
                bboxes.append([x_center, y_center, w, h])
                class_ids.append(int(class_id))

        # Apply Augmentations
        transform_input = {"image": image, "bboxes": bboxes, "class_labels": class_ids}

        if self.transform:
            augmented = self.transform(**transform_input)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            class_ids = augmented["class_labels"]

        target = {}
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(class_ids, dtype=torch.int64)

        return image, target

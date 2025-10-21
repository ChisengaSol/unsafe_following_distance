import os
import cv2
import torch
from torch.utils.data import Dataset

def convert_yolo_to_pascal_voc(x_center, y_center, w, h):
    """
    Converts a YOLO format bounding box [x_center, y_center, width, height]
    to a pascal_voc format box [xmin, ymin, xmax, ymax].
    All coordinates are normalized (0 to 1).
    """
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return [xmin, ymin, xmax, ymax]

class VehicleDataset(Dataset):
    """
    A PyTorch Dataset class to load images and their labels for object detection.
    """
    def __init__(self, image_dir, label_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        bboxes = []
        class_ids = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = [float(p) for p in parts]
                
                # Convert to pascal_voc
                pascal_voc_box = convert_yolo_to_pascal_voc(x_center, y_center, w, h)
                bboxes.append(pascal_voc_box)
                class_ids.append(int(class_id))

        transform_input = {
            'image': image,
            'bboxes': bboxes,
            'labels': class_ids
        }

        if self.transform:
            augmented = self.transform(**transform_input)
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_ids = augmented['labels']

        target = {}
        target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(class_ids, dtype=torch.int64)

        return image, target
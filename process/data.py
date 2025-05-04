from torch.utils.data import Dataset
import os
import cv2
import torch
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn

class MPIIDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.kp = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.kp)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.kp['NAME'][idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        keypoints = []
        for joint in [
            'r ankle', 'r knee', 'r hip',
            'l ankle', 'l knee', 'l hip',
            'pelvis', 'thorax', 'upper neck', 'head top',
            'r wrist', 'r elbow', 'r shoulder',
            'l wrist', 'l elbow', 'l shoulder']:
            x, y = self.kp[f'{joint}_X'][idx], self.kp[f'{joint}_Y'][idx]
            x, y = float(x), float(y)
            if x == -1 or y == -1:
                x, y = 0.0, 0.0
            keypoints.append([x / w, y / h])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, keypoints

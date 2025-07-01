


from torch.utils.data import Dataset
import os
import cv2
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import random

# class MPIIDataset(Dataset):
#     def __init__(self, csv_path, image_dir, transform=None, augment=False):
#         self.kp = pd.read_csv(csv_path)
#         self.image_dir = image_dir
#         self.transform = transform
#         self.augment = augment

#         self.joints = [
#             'r ankle', 'r knee', 'r hip',
#             'l ankle', 'l knee', 'l hip',
#             'pelvis', 'thorax', 'upper neck', 'head top',
#             'r wrist', 'r elbow', 'r shoulder',
#             'l wrist', 'l elbow', 'l shoulder'
#         ]

#     def __len__(self):
#         return len(self.kp)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.kp['NAME'][idx])
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w, _ = image.shape

#         keypoints = []
#         visibility = []

#         for joint in self.joints:
#             x, y = self.kp[f'{joint}_X'][idx], self.kp[f'{joint}_Y'][idx]
#             x, y = float(x), float(y)

#             if x == -1 or y == -1:
#                 keypoints.append([0.0, 0.0])  # zero out missing
#                 visibility.append(0)
#             else:
#                 keypoints.append([x / w, y / h])  # normalize
#                 visibility.append(1)

#         keypoints = np.array(keypoints, dtype=np.float32)
#         visibility = np.array(visibility, dtype=np.float32)

#         # Resize to 256x256
#         image = cv2.resize(image, (256, 256))

#         if self.augment:
#             image, keypoints = self.apply_augmentation(image, keypoints)

#         if self.transform:
#             image = self.transform(image)

#         keypoints = torch.tensor(keypoints, dtype=torch.float32)
#         visibility = torch.tensor(visibility, dtype=torch.float32)

#         return image, keypoints, visibility

#     def apply_augmentation(self, image, keypoints):
#         h, w, _ = image.shape

#         # Random horizontal flip
#         if random.random() > 0.5:
#             image = cv2.flip(image, 1)
#             keypoints[:, 0] = 1.0 - keypoints[:, 0]  # flip x-coordinates

#         # Random scaling (±15%)
#         scale = 1.0 + (random.random() - 0.5) * 0.3
#         keypoints = keypoints - 0.5
#         keypoints *= scale
#         keypoints = np.clip(keypoints + 0.5, 0.0, 1.0)

#         return image, keypoints


class MPIIDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, augment=False):
        self.kp = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.augment_enabled = augment

        self.joints = [
            'r ankle', 'r knee', 'r hip',
            'l ankle', 'l knee', 'l hip',
            'pelvis', 'thorax', 'upper neck', 'head top',
            'r wrist', 'r elbow', 'r shoulder',
            'l wrist', 'l elbow', 'l shoulder'
        ]  # same as before

    def __len__(self):
        return len(self.kp) * (2 if self.augment_enabled else 1)

    def apply_augmentation(self, image, keypoints):
        h, w, _ = image.shape

        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            keypoints[:, 0] = 1.0 - keypoints[:, 0]  # flip x-coordinates

        # Random scaling (±15%)
        scale = 1.0 + (random.random() - 0.5) * 0.3
        keypoints = keypoints - 0.5
        keypoints *= scale
        keypoints = np.clip(keypoints + 0.5, 0.0, 1.0)

        return image, keypoints

    def __getitem__(self, idx):
        orig_idx = idx % len(self.kp)
        apply_aug = self.augment_enabled and (idx >= len(self.kp))

        img_path = os.path.join(self.image_dir, self.kp['NAME'][orig_idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        keypoints, visibility = [], []
        for joint in self.joints:
            x, y = float(self.kp[f'{joint}_X'][orig_idx]), float(self.kp[f'{joint}_Y'][orig_idx])
            if x == -1 or y == -1:
                keypoints.append([0.0, 0.0])
                visibility.append(0)
            else:
                keypoints.append([x / w, y / h])
                visibility.append(1)

        keypoints = np.array(keypoints, dtype=np.float32)
        visibility = np.array(visibility, dtype=np.float32)

        image = cv2.resize(image, (256, 256))

        if apply_aug:
            image, keypoints = self.apply_augmentation(image, keypoints)

        if self.transform:
            image = self.transform(image)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(keypoints, dtype=torch.float32),
            torch.tensor(visibility, dtype=torch.float32),
        )

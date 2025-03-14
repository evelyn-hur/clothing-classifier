import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset

class DeepFashionSubsetDataset(Dataset):
    def __init__(self, csv_file, images_folder, transform=None):
        # csv_file (str)- path to the CSV file that has columns:
        # [image_filename, annotation_filename, category_id, category_name]
        # images_folder (str)- Path to the folder containing images
        # transform (callable, optional)- a function that takes in a PIL image
        # and returns a transformed version

        self.csv_file = csv_file
        self.images_folder = images_folder
        self.transform = transform
        self.samples = []

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_filename = row['image_filename']
                cat_id = int(row['category_id']) - 1
                self.samples.append((img_filename, cat_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns image (Tensor)- Transformed image tensor and
        # label (int)- The numeric category ID (0 through 12)
        img_filename, label = self.samples[idx]
        img_path = os.path.join(self.images_folder, img_filename)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

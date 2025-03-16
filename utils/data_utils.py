import os
import csv
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DeepFashionSubsetDataset(Dataset):
    def __init__(self, csv_file, images_folder, transform=None, use_bbox=False):
        # csv_file (str)- path to the CSV file that has columns:
        # [image_filename, annotation_filename, category_id, category_name]
        # images_folder (str)- Path to the folder containing images
        # transform (callable, optional)- a function that takes in a PIL image
        # and returns a transformed version

        self.csv_file = csv_file
        self.images_folder = images_folder
        self.transform = transform
        self.use_bbox = use_bbox
        self.samples = []

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_filename = row['image_filename']
                cat_id = int(row['category_id']) - 1
                anno_filename = row['annotation_filename']
                self.samples.append((img_filename, anno_filename, cat_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns image (Tensor)- Transformed image tensor and
        # label (int)- The numeric category ID (0 through 12)
        img_filename, anno_filename, label = self.samples[idx]
        img_path = os.path.join(self.images_folder, img_filename)

        image = Image.open(img_path).convert('RGB')

        if self.use_bbox:
            anno_path = os.path.join(self.images_folder.replace("images", "annotations"), anno_filename)
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)

            bbox = self._get_dominant_bbox(anno_data)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def _get_dominant_bbox(self, anno_data):
        # Some logic that returns the bounding box of the largest item 
        # or specifically the one matching the category in this sample
        # (In your earlier code, you had find_dominant_item.)
        # For demonstration:
        largest_area = 0
        best_bbox = None
        i = 1
        while True:
            item_key = f"item{i}"
            if item_key not in anno_data:
                break
            bbox = anno_data[item_key]["bounding_box"]
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                best_bbox = bbox
            i += 1
        return best_bbox

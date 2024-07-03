import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from sklearn.neighbors import NearestNeighbors

import datasets.dataset_utils as dataset_utils

SIZES = {
    "val_set":        [  139_104,  134],
    "synt_melbourne": [  394_632, 1249],
    "synt_paris":     [1_792_572,  268],
    "synt_berlin":    [1_285_584,  255],
}


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset_name):
        self.dataset_name = dataset_name
        self.database_folder = f"{dataset_dir}/{dataset_name}/database"
        self.queries_folder = f"{dataset_dir}/{dataset_name}/queries"
        num_db, num_q = SIZES[dataset_name]

        self.database_paths = dataset_utils.read_images_paths(
            self.database_folder, get_abs_path=True
        )
        self.queries_paths = dataset_utils.read_images_paths(
            self.queries_folder, get_abs_path=True
        )

        assert len(self.database_paths) == num_db
        assert len(self.queries_paths) == num_q

        # Read paths and UTM coordinates for all images.
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]
        ).astype(float)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
        ).astype(float)

        self.images_paths = self.database_paths + self.queries_paths

        self.num_db = len(self.database_paths)
        self.num_q = len(self.queries_paths)

        self.base_transform = tfm.Compose(
            [
                tfm.Resize(512),
                tfm.ToTensor(),
                tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def open_image(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = self.open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index, image_path

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.num_q}; #db: {self.num_db} >"

    def get_positives(self, positive_dist_threshold=25):
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        positives_per_query = knn.radius_neighbors(
            self.queries_utms, radius=positive_dist_threshold, return_distance=False
        )
        return positives_per_query

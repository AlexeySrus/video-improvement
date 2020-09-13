import cv2
import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from tqdm import tqdm
from denoising_pipeline.utils.image_utils \
    import load_image, random_crop_with_transforms
from denoising_pipeline.utils.tensor_utils import preprocess_image


class SeriesAndComputingClearDataset(Dataset):
    def __init__(self, images_series_folder, clear_images_path, window_size):
        self.series_folders_pathes = [
            os.path.join(images_series_folder, sfp)
            for sfp in os.listdir(images_series_folder)
        ]

        self.clear_series_pathes = [
            os.path.join(clear_images_path, cfp)
            for cfp in os.listdir(clear_images_path)
        ]

        assert len(self.series_folders_pathes) == len(
            self.clear_series_pathes)

        self.sort_key = lambda s: int(s.split('_')[-1].split('.')[0])

        self.series_folders_pathes.sort(key=self.sort_key)
        self.clear_series_pathes.sort(key=self.sort_key)

        self.series_folders_pathes = [
            [os.path.join(sfp, img_name) for img_name in os.listdir(sfp)]
            for sfp in self.series_folders_pathes
        ]

        self.clear_series_pathes = [
            [os.path.join(cfp, img_name) for img_name in os.listdir(cfp)]
            for cfp in self.clear_series_pathes
        ]

        for i in range(len(self.series_folders_pathes)):
            self.series_folders_pathes[i].sort(key=self.sort_key)
            self.clear_series_pathes[i].sort(key=self.sort_key)

        self.series_folders_images = []
        self.clear_series_folders_images = []

        print('Loading images into RAM:')
        for i in tqdm(range(len(self.series_folders_pathes))):
            self.series_folders_images.append(
                [
                    load_image(self.series_folders_pathes[i][j])
                    for j in range(len(self.series_folders_pathes[i]))
                ]
            )

            self.clear_series_folders_images.append(
                [
                    load_image(self.clear_series_pathes[i][j])
                    for j in range(len(self.clear_series_pathes[i]))
                ]
            )

        self.window_size = window_size

    def get_random_images(self):
        select_series_index = random.randint(
            0,
            len(self.series_folders_pathes) - 1
        )

        select_image_index = random.randint(
            0,
            len(self.series_folders_pathes[select_series_index]) - 1
        )

        select_image = self.series_folders_images[
            select_series_index][select_image_index]

        clear_image = self.clear_series_folders_images[
            select_series_index][select_image_index]

        return select_image, clear_image

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            *self.get_random_images(), self.window_size
        )

        return preprocess_image(crop1), preprocess_image(crop2)

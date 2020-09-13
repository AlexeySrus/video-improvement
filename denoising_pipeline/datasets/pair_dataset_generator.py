import os
from torch.utils.data import Dataset
import random
import numpy as np
from denoising_pipeline.utils.image_utils \
    import load_image, random_crop_with_transforms
from denoising_pipeline.utils.tensor_utils import preprocess_image


class TwoImagesDataset(Dataset):
    def __init__(self, image1_path, image2_path, window_size,):
        self.img1 = load_image(image1_path)
        self.img2 = load_image(image2_path)

        assert self.img1.shape == self.img2.shape

        self.window_size = window_size

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            self.img1,
            self.img2,
            self.window_size
        )
        crop1, crop2 = np.random.choice(
            [crop1, crop2],
            [crop2, crop1],
        )

        return preprocess_image(crop1), preprocess_image(crop2)


class PairsSeriesDataset(Dataset):
    def __init__(self, folder_path, window_size):
        images_names_list = os.listdir(folder_path)
        images_names_list.sort()

        self.images = [
            load_image(os.path.join(folder_path, img_name))
            for img_name in images_names_list
        ]

        assert len(self.images) % 2 == 0

        for i in range(0, len(self.images), 2):
            assert self.images[i].shape == self.images[i + 1].shape

        self.window_size = window_size

        self.indexes = np.array(list(range(len(self.images)))).reshape((-1, 2))

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            *self.indexes[np.random.randint(0, len(self.indexes))],
            self.window_size
        )
        crop1, crop2 = np.random.choice(
            [crop1, crop2],
            [crop2, crop1],
        )

        return preprocess_image(crop1), preprocess_image(crop2)


class SeriesAndClearDataset(Dataset):
    def __init__(self, series_folders_path, clear_images_path, window_size):
        self.series_folders_pathes = [
            os.path.join(series_folders_path, sfp)
            for sfp in os.listdir(series_folders_path)
        ]

        self.clear_images_pathes = [
            os.path.join(clear_images_path, cip)
            for cip in os.listdir(clear_images_path)
        ]

        assert len(self.series_folders_pathes) == len(
            self.series_folders_pathes)

        self.sort_key = lambda s: int(s.split('_')[-1].split('.')[0])

        self.series_folders_pathes.sort(key=self.sort_key)
        self.clear_images_pathes.sort(key=self.sort_key)

        self.series_folders_pathes = [
            [os.path.join(sfp, img_name) for img_name in os.listdir(sfp)]
            for sfp in self.series_folders_pathes
        ]

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

        select_image = load_image(
            self.series_folders_pathes[select_series_index][select_image_index]
        )

        clear_image = load_image(self.clear_images_pathes[select_series_index])

        return select_image, clear_image

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            *self.get_random_images(), self.window_size
        )

        return preprocess_image(crop1), preprocess_image(crop2)


class SeriesAndComputingClearDataset(Dataset):
    def __init__(self, series_folders_path, clear_series_path, window_size):
        self.series_folders_pathes = [
            os.path.join(series_folders_path, sfp)
            for sfp in os.listdir(series_folders_path)
        ]

        self.clear_series_pathes = [
            os.path.join(clear_series_path, cfp)
            for cfp in os.listdir(clear_series_path)
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

        select_image = load_image(
            self.series_folders_pathes[select_series_index][select_image_index]
        )

        clear_image = load_image(
            self.clear_series_pathes[select_series_index][select_image_index]
        )

        return select_image, clear_image

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            *self.get_random_images(), self.window_size
        )

        return preprocess_image(crop1), preprocess_image(crop2)

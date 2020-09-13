import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from denoising_pipeline.utils.image_utils \
    import Rotate, rotate_crop
from denoising_pipeline.utils.tensor_utils import preprocess_image


class VideoFramesGenerator:
    def __init__(self, video_source):
        self.video_source = video_source
        self.frames_count = 0
        self.video_capture = None

        self.open_video_source()

    def open_video_source(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_capture = cv2.VideoCapture(self.video_source)

        if self.video_capture is None:
            assert ValueError('Can\'t open video: {}'.format(self.video_source))

        self.video_capture.set(cv2.CAP_PROP_FOURCC, fourcc)

    def __len__(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))

    def get_resolution(self):
        return (
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )

    def get_next_frame(self):
        ret, frame = self.video_capture.read()

        frame_type = 'next'

        if not ret:
            self.video_capture.release()
            self.open_video_source()
            ret, frame = self.video_capture.read()
            if not ret:
                raise ValueError(
                    'Cant\'t read frame from reopen video source: {}'.format(
                        self.video_source
                    )
                )
            frame_type = 'new'

        return {
            'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            'type': frame_type
        }

    def __getitem__(self, idx):
        pass

    def __del__(self):
        self.video_capture.release()


class SequentialDataset(Dataset):
    def __init__(self, images_lists_path, window_size, series_size,
                 out_type='center'):
        self.image_series = [
            [
                cv2.imread(
                    images_lists_path + '/' + img_list_path + '/' + img_path,
                    1
                )
                for img_path in os.listdir(
                    images_lists_path + '/' + img_list_path
                )
            ]
            for img_list_path in os.listdir(images_lists_path)
        ]

        for i in range(len(self.image_series)):
            for j in range(len(self.image_series[i])):
                self.image_series[i][j] = cv2.cvtColor(
                    self.image_series[i][j],
                    cv2.COLOR_BGR2RGB
                )

        for images in self.image_series:
            assert len(self.image_series[0]) == len(images)
            assert len(images) == series_size + 1
            for img in images:
                assert self.image_series[0][0].shape == img.shape

        self.window_size = window_size
        self.out_type = out_type
        self.series_size = series_size + 1

        self.max_crop_shape = (
            self.image_series[0][0].shape[0] - self.window_size,
            self.image_series[0][0].shape[1] - self.window_size
        )

    def get_random_crop(self):
        series_index = np.random.randint(0, len(self.image_series))

        output_index = 0
        if self.out_type == 'center':
            output_index = self.series_size // 2

        x = np.random.randint(0, self.max_crop_shape[1])
        y = np.random.randint(0, self.max_crop_shape[0])

        return_images = []

        for i, img in enumerate(self.image_series[series_index]):
            if i == output_index:
                continue
            return_images.append(
                img[
                    y:y + self.window_size,
                    x:x + self.window_size
                ].copy()
            )

        return_images = [self.image_series[series_index][output_index][
                y:y + self.window_size,
                x:x + self.window_size
            ].copy()] + return_images

        return tuple(return_images)

    @staticmethod
    def transformation(imgs):
        select_rotation = np.random.choice(
            [
                Rotate.NONE,
                Rotate.ROTATE_90_CLOCKWISE,
                Rotate.ROTATE_180,
                Rotate.ROTATE_90_COUNTERCLOCKWISE
            ]
        )

        imgs = [
            rotate_crop(img, select_rotation)
            for img in imgs
        ]

        if np.random.rand() > 0.5:
            return imgs

        imgs = [
            cv2.flip(img, 1)
            for img in imgs
        ]

        return tuple(imgs)

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        return tuple([
            preprocess_image(img)
            for img in self.transformation(self.get_random_crop())
        ])

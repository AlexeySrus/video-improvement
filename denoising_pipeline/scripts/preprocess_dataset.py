from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from PIL import Image
import os


def parse_args():
    parser = ArgumentParser(description='Prepare dataset to train loader.')
    parser.add_argument('--dataset-path', required=True, type=str)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    noise_datasets = os.path.join(
        args.dataset_path,
        'images_series/'
    )
    clear_datasets = os.path.join(
        args.dataset_path,
        'averaged_outclass_series/'
    )

    series_folders_pathes = [
        os.path.join(noise_datasets, sfp)
        for sfp in os.listdir(noise_datasets)
    ]
    sort_key = lambda s: int(s.split('_')[-1].split('.')[0])
    series_folders_pathes.sort(key=sort_key)

    clears_series_folders_pathes = []

    for path in series_folders_pathes:
        dpath = os.path.join(clear_datasets, os.path.basename(path))
        clears_series_folders_pathes.append(dpath)
        if not os.path.isdir(dpath):
            os.makedirs(dpath)

    clears_series_folders_pathes.sort(key=sort_key)

    loop_generator = tqdm(range(len(series_folders_pathes))) \
        if args.verbose else \
        range(len(series_folders_pathes))

    for i in loop_generator:
        images_pathes = [
            os.path.join(series_folders_pathes[i], iname)
            for iname in os.listdir(series_folders_pathes[i])
            if '.DS_Store' not in iname
        ]

        for img_index in range(len(images_pathes)):
            images = np.array([
                np.array(Image.open(im).convert('RGB'))
                for k, im in enumerate(images_pathes)
                if k != img_index
            ]).astype(np.float16)

            mean_image = images.mean(axis=0)

            Image.fromarray(
                mean_image.astype(np.uint8)
            ).save(
                os.path.join(
                    clears_series_folders_pathes[i],
                    'mean_image_without_{}.png'.format(img_index)
                )
            )


if __name__ == '__main__':
    main()

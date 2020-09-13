import argparse
import cv2
from PIL import Image
import tqdm
from functools import reduce
import os
import numpy as np
import yaml
from denoising_pipeline.utils.image_matcher import ImageMatcher
from denoising_pipeline.utils.image_utils import image_preprocessing


def argument_parser():
    arg_pars = argparse.ArgumentParser(
        description='Video source test'
    )
    arg_pars.add_argument('--series-path',
                          required=True,
                          type=str
                          )
    arg_pars.add_argument('--config', required=False, type=str,
                          default='../configuration/example_train_config.yml',
                          help='Path to configuration yml file.'
                          )
    return arg_pars.parse_args()


def add_prefix(path, pref):
    """
    Add prefix to file in path
    Args:
        path: path to file
        pref: prefix

    Returns:
        path to file with named with prefix

    """
    splitted_path = list(os.path.split(path))
    splitted_path[-1] = pref + splitted_path[-1]
    return reduce(lambda x, y: x + '/' + y, splitted_path)


def load_images_from_path(path):
    names = os.listdir(path)
    return [
        np.array(Image.open(path + '/' + name).convert('RGB'))
        for name in names
    ]


if __name__ == '__main__':
    args = argument_parser()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.save = config['datasets']['images_path']
    args.match = config['datasets']['match']

    frames = load_images_from_path(args.series_path)

    print(len(frames))

    base_frame_ind = len(frames) // 2

    print('Chosen base frame id: {}'.format(base_frame_ind + 1))

    if args.save is not None:
        if not os.path.isdir(args.save):
            os.makedirs(args.save)

    if not config['datasets']['preprocessing']:
        image_preprocessing = None

    for i, frame in enumerate(tqdm.tqdm(frames)):
        if args.save is not None:
            file_path = args.save + '/{}.png'.format(i + 1)
            if args.match and i != base_frame_ind:
                Image.fromarray(
                    ImageMatcher(
                        config['datasets']['match_method'],
                        preprocessing=image_preprocessing
                    )(
                        [frames[base_frame_ind], frame]
                    )
                ).save(file_path)
            else:
                Image.fromarray(frame).save(file_path)
        else:
            if args.match:
                if i == base_frame_ind:
                    cv2.imshow('Base frame', frame)
                else:
                    cv2.imshow(
                        'Matched frame {}'.format(i + 1),
                        ImageMatcher(
                            config['datasets']['match_method'],
                            preprocessing=image_preprocessing
                        )(
                            [frames[base_frame_ind], frame]
                        )
                    )
            else:
                cv2.imshow('Frame {}'.format(i + 1), frame)

    if args.save is None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import argparse
import cv2
from PIL import Image
import tqdm
from functools import reduce
import os
import numpy as np
from denoising_pipeline.utils.image_matcher import ImageMatcher
from denoising_pipeline.datasets import VideoFramesGenerator
from denoising_pipeline.utils.image_utils import image_preprocessing


def argument_parser():
    arg_pars = argparse.ArgumentParser(
        description='Video source test'
    )
    arg_pars.add_argument('--video',
                          required=True,
                          type=str
                          )
    arg_pars.add_argument('--n', type=int, required=False, default=1)
    arg_pars.add_argument('--save', required=False, type=str,
                          help='Path to save frame series')
    arg_pars.add_argument('--match', action='store_true',
                          help='Match frames series')
    arg_pars.add_argument('--matcher-method', required=False,
                          default='ECC', choices=['ECC', 'ORB', 'ecc', 'orb'])
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


def l2_between_images(_img1, _img2):
    i1 = _img1.astype(np.float32) / 255.0
    i2 = _img2.astype(np.float32) / 255.0
    return (np.abs(i1 - i2) ** 2).sum()


if __name__ == '__main__':
    args = argument_parser()

    video_source = VideoFramesGenerator(args.video)

    print(len(video_source))

    frames = []
    for i in range(args.n):
        frames.append(video_source.get_next_frame()['frame'])

    base_frame_ind = len(frames) // 2

    print('Chosen base frame id: {}'.format(base_frame_ind + 1))

    for i, frame in enumerate(tqdm.tqdm(frames)):
        if args.save is not None:
            file_path = args.save + '/{}.png'.format(i + 1)
            if args.match and i != base_frame_ind:
                Image.fromarray(
                    ImageMatcher(
                        args.matcher_method.upper(),
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
                    cv2.imshow(
                        'Base frame',
                        frame
                    )
                else:
                    cv2.imshow(
                        'Matched frame {}'.format(i + 1),
                        ImageMatcher(
                            args.matcher_method.upper(),
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

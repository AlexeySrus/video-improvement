import argparse
import cv2
from functools import reduce
import os
import numpy as np
from denoising_pipeline.utils.image_matcher import ImageMatcher


def argument_parser():
    arg_pars = argparse.ArgumentParser(
        description='Images matcher'
    )
    arg_pars.add_argument('--first-image',
                          required=True,
                          type=str
                          )
    arg_pars.add_argument('--second-image',
                          required=True,
                          type=str
                          )
    arg_pars.add_argument('--save', action='store_true')
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
    img1 = cv2.imread(args.first_image, 1)
    img2 = cv2.imread(args.second_image, 1)

    img2_to_1 = ImageMatcher('ECC')([img1, img2])

    print('ECC:', l2_between_images(img1, img2_to_1))
    print('ORB:', l2_between_images(img1, ImageMatcher('ORB')([img1, img2])))

    if args.save:
        cv2.imwrite(add_prefix(args.second_image, 'match_'), img2_to_1)
    else:
        cv2.imshow('Window21', img2_to_1)
        cv2.imshow('Window1', img1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

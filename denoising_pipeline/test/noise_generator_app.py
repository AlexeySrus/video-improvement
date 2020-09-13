import cv2
import argparse
import os
import numpy as np
import noise


pnoise = np.vectorize(noise.pnoise1)


def generate_base(shape):
    res = np.random.rand(*shape)
    return pnoise(res)


class NoiseGenerator:
    def __init__(self, image, k=1):
        self.image = image
        self.k = k
        self.result_image = self.image.copy()
        self.need_update = True

    def set_parameter(self, x):
        if x <= 0:
            return
        if self.k != x:
            self.k = x
            self.need_update = True

    def get_noise_image(self):
        if self.need_update:
            self.result_image = ((
                    self.image.astype(
                        np.float32
                    ) / 255.0 + np.array(
                        [
                            generate_base(self.image.shape[:2]) / self.k
                            for _ in range(3)
                        ],
                        dtype=np.float32
                    ).transpose((1, 2, 0))
            ).clip(0, 1) * 255.0).astype(np.uint8)

            self.need_update = False

        return self.result_image


def argument_parser():
    arg_pars = argparse.ArgumentParser(
        description='Noise generation'
    )

    arg_pars.add_argument(
        '--input',
        required=True,
        type=str
    )

    return arg_pars.parse_args()


def main():
    args = argument_parser()

    save_image_path = os.path.join(
        os.path.dirname(args.input),
        str(os.path.basename(args.input).split('.')[0]) + '_out.png'
    )

    input_image = cv2.imread(args.input, 1)

    assert input_image is not None

    ng = NoiseGenerator(input_image)

    window_name = 'Noise generator'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(
        'Noise normed',
        window_name,
        1, 25,
        ng.set_parameter
    )

    while True:
        cv2.imshow(window_name, ng.get_noise_image())
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    main()

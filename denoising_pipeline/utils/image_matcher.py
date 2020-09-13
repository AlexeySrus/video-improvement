from denoising_pipeline.utils.auto_stack import (stackImagesECC,
                                                stackImagesKeypointMatching)


class ImageMatcher:
    support_methods = ['ECC', 'ORB']

    def __init__(self, method=None, preprocessing=None):
        """
        Constructor
        Args:
            method: Can be: ECC, ORB
        """
        self.method = 'ECC' if method is None else method.upper()
        assert self.method in self.support_methods
        self.preprocessing = preprocessing

    def __call__(self, images_list):
        assert type(images_list) is list
        assert len(images_list) > 1

        stacked_image = None
        if self.method == 'ECC':
            stacked_image = stackImagesECC(images_list, self.preprocessing)
        elif self.method == 'ORB':
            stacked_image = stackImagesKeypointMatching(images_list)

        return stacked_image


def match_images_list(images, base_image='center', match_method='ECC'):
    matcher = ImageMatcher(match_method)

    base_image_ind = 0
    if base_image == 'center':
        base_image_ind = len(images) // 2

    result = []

    for i, img in enumerate(images):
        if i == base_image_ind:
            result.append(img)
        else:
            result.append(matcher([images[base_image_ind], img]))

    return result
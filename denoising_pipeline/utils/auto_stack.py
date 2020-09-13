import os
import cv2
import numpy as np
from time import time


# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECC(file_list, preprocessing=None):
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_images = []

    for file in file_list:
        image = file.astype(np.float32) / 255
        if first_image is None:
            # convert to gray scale floating point image
            first_image = image.copy()
        else:
            # Estimate perspective transform
            if preprocessing is None:
                s, M = cv2.findTransformECC(
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY),
                    M, cv2.MOTION_HOMOGRAPHY
                )
            else:
                _img = preprocessing(
                    (image.copy() * 255.0).astype('uint8')
                ).astype(np.float32) / 255
                _first_img = preprocessing(
                    (first_image.copy() * 255.0).astype('uint8')
                ).astype(np.float32) / 255

                s, M = cv2.findTransformECC(
                    cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(_first_img, cv2.COLOR_RGB2GRAY),
                    M, cv2.MOTION_HOMOGRAPHY
                )
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            stacked_images.append(image)

    first_image = file_list[0]
    stacked_images.sort(key=lambda img: np.abs(first_image - img).sum())

    return (stacked_images[0] * 255).astype(np.uint8)


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(file_list):
    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_images = []
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        image = file
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_images.append(imageF)

    first_image = file_list[0]
    stacked_images.sort(key=lambda img: np.abs(first_image - img).sum())

    return (stacked_images[0] * 255).astype(np.uint8)
"""
This module provides methods, to perform affine transformations on images to augment a data set.
Methods have been implemented as proposed by Sebastian Sudholt.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import numpy as np
import sys
try:
    import cv2
except ImportError:
    sys.path.append('/vol/local/install/opencv-4.1.0/lib/python3.5/dist-packages')
    import cv2


def homography_augm(img, random_limits=(0.9, 1.1)):
    """
    Creates an augmentation by computing a homography from three
    points in the image to three randomly generated points
    """
    y, x = img.shape[:2]
    fx = float(x)
    fy = float(y)
    src_point = np.float32([[fx/2, fy/3,],
                            [2*fx/3, 2*fy/3],
                            [fx/3, 2*fy/3]])
    random_shift = (np.random.rand(3,2) - 0.5) * 2 * (random_limits[1]-random_limits[0]) / 2 \
                   + np.mean(random_limits)
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if img.ndim == 3:
        border_value = np.median(np.reshape(img, (img.shape[0]*img.shape[1], -1)), axis=0)
    else:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(x,y), borderValue=float(border_value))
    return warped_img


def scale(img, w=None, h=None):
    """
    Scaling of an image.
    If width and height are not defined, the image is not transformed.
    If only one is defined, the original ratio is kept while resizing.
    If both are defined, the image is resized to the specified width and height.

    :param img: image to be scaled
    :param w: new width
    :param h: new height
    :return: scaled image
    """
    if w is None and h is None:
        return img
    elif w is not None and h is None:
        ratio = img.shape[0] / img.shape[1]
        shape = (w, ratio * w)
    elif w is None and h is not None:
        ratio = img.shape[1] / img.shape[0]
        shape = (ratio * h, h)
    else:
        shape = (w, h)
    shape = tuple(map(int, shape))
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    return img


def visualiz_homography_augm(img, random_limits=(0.9, 1.1)):
    """
    Creates an augmentation by computing a homography from three
    points in the image to three randomly generated points
    """
    y, x = img.shape[:2]
    fx = float(x)
    fy = float(y)
    src_point = np.float32([[fx/2, fy/3,],
                            [2*fx/3, 2*fy/3],
                            [fx/3, 2*fy/3]])
    random_shift = (np.random.rand(3,2) - 0.5) * 2 * (random_limits[1]-random_limits[0]) / 2 \
                   + np.mean(random_limits)
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if img.ndim == 3:
        border_value = np.median(np.reshape(img, (img.shape[0]*img.shape[1], -1)), axis=0)
    else:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(x,y), borderValue=float(border_value))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    # original image
    ax = plt.subplot(121)
    plt.imshow(img, cmap='bone')
    for p in src_point:
        circ = Circle(p, 2, fill=False, color='r')
        ax.add_patch(circ)
    plt.axis('off')
    # warped image
    ax = plt.subplot(122)
    plt.imshow(warped_img, cmap='bone')
    for p in dst_point:
        circ = Circle(p, 2, fill=False, color='r')
        ax.add_patch(circ)
    plt.axis('off')

    plt.show()

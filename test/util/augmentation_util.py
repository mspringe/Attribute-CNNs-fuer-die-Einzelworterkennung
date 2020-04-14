from unittest import TestCase
import matplotlib.pyplot as plt
import numpy as np
from src.util import augmentation_util


class TestModule(TestCase):

    def setUp(self):
        self.img = np.zeros((500,1000), dtype=float)
        for i,j in zip(np.arange(0,500,20), np.arange(20,520,20)):
            self.img[i:j,i:j] = 1.
            self.img[i:j,i+500:j+500] = 1.

    def show(self, img):
        plt.imshow(img, cmap='jet')
        plt.show()

    def test_img(self):
        self.show(self.img)

    def test_augment(self):
        img_aug = augmentation_util.homography_augm(self.img)
        self.show(img_aug)

    def test_scale(self):
        self.show(self.img)
        img_res = augmentation_util.scale(self.img, w=200, h=50)
        self.show(img_res)
        img_res = augmentation_util.scale(self.img, h=50)
        self.show(img_res)
        img_res = augmentation_util.scale(self.img, w=50)
        self.show(img_res)

    def test_visualiz_homography_augm(self):
        augmentation_util.visualiz_homography_augm(self.img)

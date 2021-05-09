# -*- coding: utf-8 -*-

"""
Date                :  12/08/2017
Author              :  Adrián Homero Moreno García
Objective of Script :  This script has the main function that will call the classes
                       for make the process of the TopHatTransformation
"""


import numpy as np
import cv2

from auxiliarymodules.BasicOperations import BasicOperations
from auxiliarymodules.MorphologicalOperations import MorphologicalOperations
from auxiliarymodules.StructuringElements import StructuringElements


class TopHatTransform:
    # Global variables
    img_inverted = [] # List of values of the inverted image: list
    img_opening = [] # List of values of the opening gray image: list
    img_substract = []

    def generates_top_hat_transform(self, img_original, se_name, se_params):
        """
        Function that returns an image with a TopHat applied transform.
        :param img_original: Image to transform
        :param se_name: Name of the structuring element that will be used.
        :param se_params: Size of the structuring element that will be used.
        :return: Return the image resultant
        """
        global img_opening, img_inverted

         # -- STEPS OF TOP HAT TRANSFORM
        img_inverted = BasicOperations().invert_image(img_original)
        structuring_element = StructuringElements().create_structuring_element(se_name, se_params)
        img_opening = MorphologicalOperations().gray_opening_image(img_inverted, structuring_element)
        img_substract = BasicOperations().substract_imges(img_inverted, img_opening)


        for i in range(len(img_substract)):
            img_substract[i] = ((img_substract[i] - img_substract.min()) / (img_substract.max() - img_substract.min()))*255

        # Filtering Removal Noise, before Thresholding
        image_255 = np.array(img_substract * 255, dtype=np.uint8)
        gaussian_filtering_blur = cv2.GaussianBlur(image_255, (3, 5), 0)
        median_filtering_blur = cv2.medianBlur(image_255,7)

        backtorgb = cv2.cvtColor(image_255, cv2.COLOR_GRAY2RGB)

        return backtorgb

# -*- coding: utf-8 -*-

"""
Date                :  28/04/2021
Author              :  Adrián Homero Moreno García
Objective of Script :  This scrpt has the main function that will call the classes
                       for make the complete process for segmented image.
"""

from auxiliarymodules.ManipulationImage import ManipulationImage
from vesselenhancement.TopHatTransform import TopHatTransform
from vesselenhancement.SegmentationByGradient import SegmentationByGradient

class VesselEnhancementSegmentation:


    def main():
        # ------------------------------------------------------------------------------------------------------------------#
        """
        USER INPUT:
        Here are the variables that the user can put on this program.
        *************************************************************
        se_name options are: diamond, cross, elliptical
        se_params options are: if it is diamond = dict(radius=n), and cross, elliptical= dict(width=n, height=n)
        """
        img_path = 'images/inputs/9.png'
        img_gt_path = 'images/outputs/9_gt.png'
        img_th_path = 'images/outputs/9_th.png'
        se_name = 'elliptical'
        #se_params = dict(radius=22)
        se_params = dict(width=11, height=15)
        #------------------------------------------------------------------------------------------------------------------#
        # 1.- It reads the original image
        img_original = ManipulationImage().read_image(img_path, 'L')
        # 2.- It applies to it the TopHat Transform
        vessel_enhancement_image = TopHatTransform().generates_top_hat_transform(img_original, se_name, se_params)
        # 3.- Saving TopHat image
        ManipulationImage().save_image(vessel_enhancement_image, img_path)
        # 4.- Generate Segmented Image By Kohonen, Tuning A,B parameters for Kohonen Algorithm
        SegmentationByGradient().applySegmentationByGradient(img_th_path,img_gt_path)



VesselEnhancementSegmentation.main()
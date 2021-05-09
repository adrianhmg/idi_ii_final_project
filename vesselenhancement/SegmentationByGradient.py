import cv2
import numpy as np
from PIL import Image as IM
from auxiliarymodules.AreaOfInterest import AreaOfInterest

from vesselenhancement.ApplyKohonenSegmentation import applyKohonen


def error_min_cuad(x, y, a, b):
    return np.mean(((a + b * x) - y) ** 2) / 2


class SegmentationByGradient:


    def applySegmentationByGradient(self,tophat_path,gt_path):
        topHatImage = IM.open('C:/Users/Adrian_Moreno/PycharmProjects/IDI_II_Project/'+tophat_path)

        gtImage = IM.open('C:/Users/Adrian_Moreno/PycharmProjects/IDI_II_Project/'+gt_path)
        gtImage = cv2.cvtColor(np.float32(gtImage), cv2.COLOR_GRAY2RGB)
        y = np.array(gtImage)

        # Initial Values to be Optimize
        a = 28.5
        b = 35.5
        eta = 0.00001

        topHatImage_c = np.copy(topHatImage)

        lista_errores = []
        error = 0
        error_epochs = 1000000
        epochs = 0

        # Descendent Gradient
        while abs(error_epochs) > (10 ** (-1)):
            topHatImage = np.copy(topHatImage_c)
            # a - Thresholding Value
            # b - Lim Max for Centroids Creation
            x = applyKohonen(topHatImage, a, b)
            epochs += 1
            error_ant = error
            error = error_min_cuad(x, y, a, b)
            error_epochs = error_ant - error
            print(abs(error_epochs))
            lista_errores.append(error)

            a -= eta * np.mean(a + b * x - y)
            b -= eta * np.mean((a + b * x - y) * x)

            # For np.randint cannot be negative
            if b < 2:
                break

        gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        aoi = AreaOfInterest().generates_area_opening(gray_image)

        nueva = IM.fromarray(np.uint8(aoi))
        nueva.save('C:/Users/Adrian_Moreno/PycharmProjects/IDI_II_Project/images/outputs/9-tophat-kohonen-grad.png')

import numpy as np
from source.utils import *


class GaborGradScale():

    def __init__(self, img:np.ndarray, target_brightness):
        """some txt
        Args:
            img: 2d np.ndarray HxW"""
        self.img = img
        self.target_brightness = target_brightness
        self.br_templates: list = []
        self.contours: np.ndarray
        self.current_template: float

    def intensity_function(self, pixel, template):
        """some txt"""
        alpha = 0.5
        eps = 1e-6
        return np.uint8(np.abs(pixel - template))
    
    def merge_contours(self):
        """some txt"""
        combined_countours = self.contours[0]
        for contour in combined_countours[1:]:
            combined_countours += contour
        
        combined_countours = np.where(combined_countours > 0, 1, 0)
        self.contours = combined_countours

    def extract_contours(self):
        """some txt"""
        br_templates = [self.target_brightness, find_most_common_brightness(self.img)]
        self.br_templates = br_templates

        contours = np.zeros(len(br_templates), dtype=object)
        for i, template in enumerate(self.br_templates):
            self.current_template = template
            partial_contours = self.calculate_contour_points()
            contours[i] = np.array(partial_contours)

        self.contours = contours

        self.merge_contours()
        
        return self.contours
    
    def grad_f(self, image, point): 
        """calculates gradient by definition
        Args:
            image: self.image with applied intensity function to it
        """
        gx, gy = np.gradient(image)
        return [gx[point[0], point[1]], gy[point[0], point[1]]]
          
    def calculate_contour_points(self):
        """some txt"""
        const1 = 5
        const2 = -2
        eps = 1e-6
        contour = np.zeros_like(self.img)
        delta_deg = 10

        vectorized_int_func = np.vectorize(self.intensity_function)
        temp_image = vectorized_int_func(self.img, self.current_template)

        gradients = [compute_derivatives(temp_image, kernel_size = 1), 
                     compute_derivatives(temp_image, kernel_size = 3)]

        for i, row in enumerate(temp_image):
            for j, _ in enumerate(row):
                grad_ok = 0

                for grads in gradients:

                    grad_x, grad_y, grad_xx, grad_yy, grad_xxx, grad_yyy = grads
                    grad_p = [grad_x[i, j], grad_y[i, j]]
                    # считаем модуль вектора градиента (максимальная скорость изменения)
                    mag_grad = np.sqrt(grad_p[0]**2 + grad_p[1]**2)
                    angle_grad = np.arctan(grad_p[0] / (grad_p[1] + eps))

                    angle_grad_degree = np.rad2deg(angle_grad)
                    if angle_grad_degree < 0:  
                        angle_grad_degree += 360

                    dir_cos1 = grad_p[0] / (mag_grad + eps)
                    dir_cos2 = grad_p[1] / (mag_grad + eps)

                    if ((angle_grad_degree < delta_deg) and (angle_grad_degree > 360 - delta_deg)) or \
                       ((angle_grad_degree < 180 + delta_deg) and (angle_grad_degree > 180 - delta_deg)):     
                        grad_yy_delta_minus = grad_yy[i, j]
                        grad_yy_delta_plus = grad_yy[i, j]
                    else:
                        grad_yy_delta_plus = grad_yy[i, min(j + 1, temp_image.shape[1] - 1)]
                        grad_yy_delta_minus = grad_yy[i, max(j - 1, 0)]

                    if ((angle_grad_degree < 90 + delta_deg) and (angle_grad_degree > 90 - delta_deg)) or \
                       ((angle_grad_degree < 270 + delta_deg) and (angle_grad_degree > 270 - delta_deg)):
                        grad_xx_delta_minus = grad_xx[i, j]
                        grad_xx_delta_plus = grad_xx[i, j]
                    else:
                        grad_xx_delta_plus = grad_xx[min(i + 1, temp_image.shape[0] - 1), j]
                        grad_xx_delta_minus = grad_xx[max(i - 1, 0), j]

                    dir_ddf_plus = grad_xx_delta_plus*dir_cos1 + grad_yy_delta_plus*dir_cos2
                    dir_ddf_minus = grad_xx_delta_minus*dir_cos1 + grad_yy_delta_minus*dir_cos2

                    dir_dddf = grad_xxx[i, j]*dir_cos1 + grad_yyy[i, j]*dir_cos2     

                    if (mag_grad > const1) and ((dir_ddf_minus * dir_ddf_plus) < 0) and (dir_dddf < const2):
                        grad_ok += 1
                
                if grad_ok == len(gradients):
                    contour[i, j] += 1
                
        return contour
            



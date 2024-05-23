import cv2
from PIL import Image
from source.pipelines import OnnHyperPipeline
from source.GaborGradScale import GaborGradScale
from source.utils import *


def test_sobel_op(onn_pipeline_fully_set: OnnHyperPipeline):
    test_img = onn_pipeline_fully_set.dataset.samples[0].band_img[0]

    grad_x = cv2.Sobel(test_img, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(test_img, cv2.CV_64F, 0, 1, ksize=1)

    grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=1)
    grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=1)

    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results', 300, 300)  # Установка нового размера окна

    magnitude = cv2.magnitude(grad_x, grad_y)

    # Нормализация для отображения
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    # Отображение результатов
    cv2.imshow('Results', magnitude)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_contour_extr_method_1():

    image_path = r'tests\test_images\simple_scene.jpg' 
    img = Image.open(image_path)

    img_array = np.array(img, dtype=np.uint8)

    img_array = np.transpose(img_array, (2, 0, 1))

    cont_extr_module = GaborGradScale(img=np.uint8(img_array[0]), 
                                      target_brightness=255)
    
    grad_x = cv2.Sobel(np.uint8(img_array[0]), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(np.uint8(img_array[0]), cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)

    # Нормализация для отображения
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)
    
    contours = cont_extr_module.extract_contours()

    cv2.namedWindow('Results_dummy', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results_dummy', 1200, 300)  # Установка нового размера окна

    contours = np.where(contours == 1, 255 ,0)
    # contours = cv2.normalize(contours, None, 0, 255, cv2.NORM_MINMAX)
    contours = np.uint8(contours)

    combined_image = cv2.hconcat([np.uint8(img_array[0]),
                                  contours, 
                                  magnitude])

    cv2.imshow('Results_dummy', combined_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_contour_extr_method_2(onn_pipeline_fully_set: OnnHyperPipeline):
    test_img = onn_pipeline_fully_set.dataset.samples[0].band_img[0]

    cont_extr_module = GaborGradScale(img=test_img, 
                                      target_brightness=onn_pipeline_fully_set.dataset.samples[0].target_brightness[0])
    
    contours = cont_extr_module.extract_contours()

    cv2.namedWindow('Results_test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results_test', 800, 400)  # Установка нового размера окна

    contours = contours

    contours = np.where(contours == 1, 255 ,0)
    # contours = cv2.normalize(contours, None, 0, 255, cv2.NORM_MINMAX)
    contours = np.uint8(contours)

    cv2.imshow('Results_test', contours)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

import numpy as np
from PIL import Image
from source.onn_model import *
from source.utils import display_image
from source.pipelines import OnnHyperPipeline


def test_segm_module(onn_pipeline_fully_set: OnnHyperPipeline):
    """some txt"""
    print("______BEGIN______")
    ext_cont_module = OnnContourExtractionModule("ContExtr")
    contours = ext_cont_module.run(img = onn_pipeline_fully_set.dataset.samples[0].band_img[0], 
                                   find_cont_method="simple_sobel",
                                   draw_contours=False)

    segm_module = OnnSegmentationModule("Segmentation")

    print("Cont_extr_done")

    res_img = segm_module.run(img = onn_pipeline_fully_set.dataset.samples[0].band_img[0],
                              gt = onn_pipeline_fully_set.dataset.samples[0].labels,
                              contours=contours,
                              w1 = 30,
                              alpha = 0.2,
                              beta = 0.3,
                              w4 = 6,
                              threshold = 15,
                              increase_value = 0.1,
                              max_number_of_iters=35)
    
    # cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Results', 300, 300)  # Установка нового размера окна

    # cv2.imshow('Results', res_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    print(res_img)
    orig_img = onn_pipeline_fully_set.dataset.samples[0].band_img[0]
    min_val = np.min(orig_img)
    max_val = np.max(orig_img)
    scaled_orig_img = 255 * (orig_img - min_val) / (max_val - min_val)

    cv2.imwrite("original_img.png", scaled_orig_img)
    cv2.imwrite("Contours.png", contours)
    cv2.imwrite("result_segm.png", res_img)

    assert np.all(res_img < 256)
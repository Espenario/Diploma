import numpy as np
from PIL import Image
from source.onn_model import *
from source.utils import display_image
from source.pipelines import OnnHyperPipeline


def test_segm_module(onn_pipeline_fully_set: OnnHyperPipeline):
    """some txt"""
    ext_cont_module = OnnContourExtractionModule("ContExtr")
    contours = ext_cont_module.run(img = onn_pipeline_fully_set.dataset.samples[0].band_img[0], 
                                   find_cont_method="simple_sobel",
                                   draw_contours=True)

    segm_module = OnnSegmentationModule("Segmentation")

    res_img = segm_module.run(img = onn_pipeline_fully_set.dataset.samples[0].band_img[0],
                              gt = onn_pipeline_fully_set.dataset.samples[0].labels,
                              contours=contours)
    
    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results', 300, 300)  # Установка нового размера окна

    cv2.imshow('Results', res_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    assert np.all(res_img < 256)
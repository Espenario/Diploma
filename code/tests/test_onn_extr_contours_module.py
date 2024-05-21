import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnContourExtractionModule
from source.utils import find_contour_area


def test_extr_cont_library(onn_pipeline_fully_set: OnnHyperPipeline):
    """some txt"""
    ext_cont_module = OnnContourExtractionModule("ContExtr")
    ext_cont_module.run(onn_pipeline_fully_set.dataset.samples[0].band_img[0])
    assert len(ext_cont_module.contours) > 0

def test_find_contour_area_1():
    simple_contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cont_area = find_contour_area(simple_contour)

    assert cont_area == 1

def test_find_contour_area_2():
    simple_contour = np.array([[0, 0], [1, 0], [1, 1]])
    cont_area = find_contour_area(simple_contour)

    assert cont_area == 0.5


def test_find_contour_area_3(onn_pipeline_fully_set: OnnHyperPipeline):
    test_img = onn_pipeline_fully_set.dataset.samples[0].band_img[0]
    ext_cont_module = OnnContourExtractionModule("ContExtr")
    ext_cont_module.extract_cont_library(test_img)

    max_cont_area = test_img.shape[0] * test_img.shape[1]
    min_cont_area = 0

    test_sample_cont_area = find_contour_area(ext_cont_module.contours[0])

    print(ext_cont_module.contours[0])
    print(test_sample_cont_area)

    assert max_cont_area >= test_sample_cont_area
    assert min_cont_area <= test_sample_cont_area




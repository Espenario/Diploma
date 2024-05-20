import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnContourExtractionModule
from source.utils import show_contours


def test_extr_cont_library(onn_pipeline_fully_set: OnnHyperPipeline):
    """some txt"""
    ext_cont_module = OnnContourExtractionModule("ContExtr")
    ext_cont_module.run(onn_pipeline_fully_set.dataset.samples[0].band_img[0])
    assert len(ext_cont_module.contours) > 0
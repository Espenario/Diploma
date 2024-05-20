import pytest
import numpy as np
import spectral
import matplotlib.pyplot as plt
from source.pipelines import OnnHyperPipeline


def test_simple_opt(onn_pipeline_basic: OnnHyperPipeline):
    onn_pipeline_basic.select_chanels(method = 'simple_opt', n = 4)
    assert len(onn_pipeline_basic.dataset.selected_bands) == 4

def test_build_dataset(onn_pipeline_basic: OnnHyperPipeline):
    onn_pipeline_basic.select_chanels(method = 'simple_opt', n = 4)
    onn_pipeline_basic.create_samples()
    assert onn_pipeline_basic.dataset.samples[0].band_img.shape[0] == 4

def test_build_sample_part_of_image(onn_pipeline_basic: OnnHyperPipeline):
    begin_img = onn_pipeline_basic.dataset.data[0]

    bands_to_select = [1, 2, 3]

    img_to_show = spectral.get_rgb(begin_img, bands_to_select)
    img_to_show /= np.max(img_to_show)
    img_to_show = np.asarray(255 * img_to_show, dtype='uint8')

    fig, ax = plt.subplots(3, 1)

    ax[0].imshow(img_to_show)
    ax[0].set_title("from base img")

    bands_to_exclude = list(set(bands_to_select).symmetric_difference(np.arange(0, begin_img.shape[2])))

    samples_bands = begin_img[15:115, 113:213, :]
    samples_bands = np.asarray(samples_bands)

    samples_bands = np.delete(samples_bands, bands_to_exclude, axis = 2)
    samples_bands = samples_bands.transpose((2, 0, 1))

    samples_band = samples_bands[0]

    ax[1].imshow(samples_band)
    ax[1].set_title("after sampling")

    samples_bands = np.asarray([samples_band])

    samples_bands = samples_bands.transpose((1, 2, 0))

    sample_img = spectral.get_rgb(samples_bands, [0])
    sample_img /= np.max(sample_img)
    sample_img = np.asarray(255 * sample_img, dtype='uint8')

    ax[2].imshow(sample_img)
    ax[2].set_title("after sampling")

    plt.show()

    

    


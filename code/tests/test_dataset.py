import pytest
from source.pipelines import OnnHyperPipeline


def test_simple_opt(onn_pipeline: OnnHyperPipeline):
    onn_pipeline.select_chanels(method = 'simple_opt', n = 4)
    assert len(onn_pipeline.dataset.selected_bands) == 4

def test_build_dataset(onn_pipeline: OnnHyperPipeline):
    onn_pipeline.select_chanels(method = 'simple_opt', n = 4)
    onn_pipeline.create_samples()
    print(onn_pipeline.dataset.samples.shape)
    assert onn_pipeline.dataset.samples.shape[1] == 4
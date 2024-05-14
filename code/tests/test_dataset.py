import pytest
from source.pipelines import OnnHyperPipeline
import source.datasets


def test_simple_opt(onn_pipeline: OnnHyperPipeline):
    onn_pipeline.select_chanels(method = 'simple_opt', n = 4)
    assert len(onn_pipeline.dataset.selected_bands) == 4

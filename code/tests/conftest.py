import pytest
from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel2D, OnnSelectiveAttentionModule2D

@pytest.fixture(scope="session")
def onn_pipeline_basic():
    onn_pipe = OnnHyperPipeline()
    onn_pipe.add_dataset(dataset_name='PaviaU')
    onn_pipe.specify_target_class(target_class='Asphalt')
    return onn_pipe

@pytest.fixture(scope="session")
def onn_pipeline_fully_set(onn_pipeline_basic: OnnHyperPipeline):
    onn_pipeline_basic.select_chanels()
    onn_pipeline_basic.create_samples()
    return onn_pipeline_basic

@pytest.fixture(scope="session")
def onn_model_attention_only():
    model = OnnModel2D(model_name="attention_only")
    model.add_module(OnnSelectiveAttentionModule2D("SelectiveAtt"))
    return model

@pytest.fixture(scope="session")
def onn_sel_att_module(onn_pipeline_fully_set: OnnHyperPipeline):
    sample_img = onn_pipeline_fully_set.dataset.samples[0]
    sample_band = sample_img[0]
    module = OnnSelectiveAttentionModule2D("SelectiveAtt")
    module.setup_oscillators(sample_band)
    return module
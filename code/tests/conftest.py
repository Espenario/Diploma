import pytest
from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel2D, OnnSelectiveAttentionModule2D

@pytest.fixture(scope="session")
def onn_pipeline():
    onn_pipe = OnnHyperPipeline()
    onn_pipe.add_dataset(dataset_name='PaviaU')
    onn_pipe.specify_target_class(target_class='Asphalt')
    return onn_pipe

@pytest.fixture(scope="session")
def onn_model_attention_only():
    model = OnnModel2D(model_name="attention_only")
    model.add_module(OnnSelectiveAttentionModule2D("SelectiveAtt"))
    return model
import pytest
from source.pipelines import OnnHyperPipeline

@pytest.fixture(scope="session")
def onn_pipeline():
    onn_pipe = OnnHyperPipeline()
    onn_pipe.add_dataset(dataset_name='PaviaU')
    onn_pipe.specify_target_class(target_class='Asphalt')
    return onn_pipe
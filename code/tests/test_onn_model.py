from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel

def test_selective_attention_module(onn_pipeline: OnnHyperPipeline,
                                    onn_model_attention_only: OnnModel):
    """some txt"""
    model = onn_model_attention_only
    onn_pipeline.add_model(model)
    onn_pipeline.run(target_class = 'Asphalt', method="expert")
    assert type(onn_pipeline.model.segmented_samples) == dict
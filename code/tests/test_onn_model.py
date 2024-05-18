from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel


def test_selective_attention_module_run(onn_pipeline_basic: OnnHyperPipeline,
                                        onn_model_attention_only: OnnModel):
    """some txt"""
    model = onn_model_attention_only
    onn_pipeline_basic.add_model(model)
    onn_pipeline_basic.run(target_class = 'Asphalt', band_sel_method="expert")
    assert isinstance(onn_pipeline_basic.model.segmented_samples, list)
    assert isinstance(onn_pipeline_basic.model.segmented_samples[0], dict)

def test_selective_attention_module_eval(onn_pipeline_basic: OnnHyperPipeline,
                                         onn_model_attention_only: OnnModel):
    """some txt"""
    model = onn_model_attention_only
    onn_pipeline_basic.add_model(model)
    onn_pipeline_basic.run(target_class = 'Asphalt', band_sel_method="expert")
    res = onn_pipeline_basic.eval(metric = "iou")
    assert isinstance(res, list)
    assert isinstance(res[0], list)

from source.pipelines import OnnHyperPipeline
from source.onn_model import *

def main():
    """some txt"""
    onn_pipe = OnnHyperPipeline()

    onn_pipe.add_dataset(dataset_name='PaviaU')

    model = OnnModel2D(model_name="3_module_onn")
    model.add_module(OnnSelectiveAttentionModule2D("SelectiveAtt"))
    model.add_module(OnnContourExtractionModule("ContourExtr"))
    model.add_module(OnnSegmentationModule("Segmentation"))

    onn_pipe.add_model(model)

    onn_pipe.run(target_class='Asphalt',
                 optimize_before_run=True)
    onn_pipe.eval(metric = "iou")

    onn_pipe.show_results()


if __name__ == "__main__":
    main()
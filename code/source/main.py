from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel2D, OnnSelectiveAttentionModule2D, OnnContourExtractionModule


def main():
    """some txt"""
    onn_pipe = OnnHyperPipeline()

    onn_pipe.add_dataset(dataset_name='PaviaU')

    model = OnnModel2D(model_name="attention+cont_extr")
    model.add_module(OnnSelectiveAttentionModule2D("SelectiveAtt"))
    model.add_module(OnnContourExtractionModule("ContourExtr"))

    onn_pipe.add_model(model)

    onn_pipe.run(target_class='Asphalt',
                 optimize_before_run=False, 
                 params_path=r".\hyperparams\attention+cont_extr_4.json")
    onn_pipe.eval(metric = "iou")

    onn_pipe.show_results()


if __name__ == "__main__":
    main()
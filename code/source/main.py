from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel2D, OnnSelectiveAttentionModule2D


def main():
    """some txt"""
    onn_pipe = OnnHyperPipeline()

    onn_pipe.add_dataset(dataset_name='PaviaU')

    model = OnnModel2D(model_name="attention_only")
    model.add_module(OnnSelectiveAttentionModule2D("SelectiveAtt"))

    onn_pipe.add_model(model)

    onn_pipe.run(target_class='Asphalt', band_sel_method='simple_opt')
    res = onn_pipe.eval(metric = "iou")

    onn_pipe.show_results()


if __name__ == "__main__":
    main()
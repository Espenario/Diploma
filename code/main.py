from pipelines import OnnHyperPipeline
from onn_model import OnnModel


def main():
    """some txt"""
    onn_pipe = OnnHyperPipeline()

    onn_pipe.add_dataset(dataset_name='PaviaU')

    onn_pipe.add_model(OnnModel())
    
    onn_pipe.specify_target_class(target_class='Asphalt')
    onn_pipe.select_chanels(method = 'simple_opt')
    # onn_pipe.run()


if __name__ == "__main__":
    main()
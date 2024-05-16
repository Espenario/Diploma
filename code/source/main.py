from source.pipelines import OnnHyperPipeline
from source.onn_model import OnnModel


def main():
    """some txt"""
    onn_pipe = OnnHyperPipeline()

    onn_pipe.add_dataset(dataset_name='PaviaU')

    onn_pipe.add_model(OnnModel())

    onn_pipe.run(target_class='Asphalt', method = 'simple_opt')
    print(onn_pipe.result)


if __name__ == "__main__":
    main()
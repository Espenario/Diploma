from pipelines import OnnHyperPipeline


def main():
    """some txt"""
    onn_pipe = OnnHyperPipeline(dataset_name='PaviaU')
    onn_pipe.load_and_prepare_data()
    # onn_pipe.get_data_info()
    onn_pipe.dataset.view_data()
    onn_pipe.dataset.create_samples()


if __name__ == "__main__":
    main()
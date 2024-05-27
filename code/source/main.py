import argparse
from source.pipelines import OnnHyperPipeline
from source.onn_model import *

def main():
    """some txt"""
    parser = argparse.ArgumentParser(description='Программа для анализа гиперспектральных изображений с помощью осцилляторной нейронной сети')

    # Добавление аргументов
    parser.add_argument('--dataset_name', 
                        type=str, 
                        choices=['PaviaU', 'PaviaC', 'KSC', 'IndianPines', 'Botswana'],
                        default='PaviaU',
                        help='Используемый датасет (для загрузки нового понадобиться подключить vpn)')
    parser.add_argument('--hyperparams_path', 
                        type=str, 
                        default=r".\hyperparams\model_baseline.json",
                        help='путь до файла с гиперпараметрами')
    parser.add_argument('--optimize_before_run', 
                        type=bool, 
                        default=False,
                        help='запуск программы с оптимизацией гиперпараметров')
    parser.add_argument('--metric', 
                        type=str, 
                        default='iou',
                        choices=['iou', 'pixelwise'],
                        help='метрика, по которой будет оценено качество сегментации')
    parser.add_argument('--segm_output_dir', 
                        type=str, 
                        default="./segm_results",
                        help='путь до директории с результатами сегментации')
    
    args = parser.parse_args()

    onn_pipe = OnnHyperPipeline()

    onn_pipe.add_dataset(dataset_name=args.dataset_name)

    model = OnnModel2D(model_name="3_module_onn")
    model.add_module(OnnSelectiveAttentionModule2D("SelectiveAtt"))
    model.add_module(OnnContourExtractionModule("ContourExtr"))
    model.add_module(OnnSegmentationModule("Segmentation"))

    onn_pipe.add_model(model)

    onn_pipe.run(target_class='Asphalt',
                 params_path=args.hyperparams_path,
                 optimize_before_run=args.optimize_before_run)
    onn_pipe.eval(metric=args.metric)

    onn_pipe.show_results(output_file=args.segm_output_dir)


if __name__ == "__main__":
    main()
"""some module docstrings"""
from copy import deepcopy
import random
import os
import cv2
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict
import sklearn.model_selection
import seaborn as sns
import spectral
import matplotlib.pyplot as plt
from scipy import io, misc
import imageio
import torch
from source.stimul import Stimul, Sample


HYPERPARAMS = {
    "Dataset": ["num_samples", "sample_height", "sample_width", "threshold", 
                "band_sel_method", "num_of_bands"],
    "OnnSelAtModule": ["params_sel_method", "stimuls_num", "stimuls_sel_method",
                       "po_num"],
    "OnnContExtrModule": ["find_cont_method", "draw_contours", "cont_area_threshold_percent",
                          "level_value"]
}

DEFAULT_HYPERPARAMS = {
    "band_sel_method": "expert", 
    "num_samples": 3, 
    "sample_height": 100, 
    "sample_width": 100, 
    "threshold": 100, 
    "num_of_bands": 4,
    "po_num": 2,
    "params_sel_method": "expert", 
    "stimuls_num": 2, 
    "stimuls_sel_method": "brightness",
    "find_cont_method": "library", 
    "draw_contours": False, 
    "level_value": None,
    "cont_area_threshold_percent": 1,   
}

HYPERPARAMS_DIR = r"./hyperparams"

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class Result():

    def __init__(self, best_band_id, sample, best_score):
        self.best_band_id = best_band_id
        self.sample = sample
        self.best_score = best_score

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))
    
def are_close(a, b, rel_tol=1e-9, abs_tol=0.0):
    """Check if two numbers are close within a given tolerance."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape([1, 1, 3]), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

def display_image(img, rgb_bands, other_bands = []):
    """Display the specified dataset.
    Args:
        img: 3D hyperspectral image
        bands: tuple of bands to select
    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, rgb_bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*rgb_bands)
    
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title(caption)

    for i, band_id in enumerate(other_bands):
        band_img = spectral.get_rgb(img, [band_id])
        ax[(i + 1) // 2, (i + 1) % 2].imshow(band_img)
        ax[(i + 1) // 2, (i + 1) % 2].set_title(band_id)

    fig.suptitle("RBG + 3 other bands")
    plt.show()

def explore_spectrums(img, complete_gt, class_names,
                      ignored_labels=[]):
    """Plot sampled spectrums with mean + std for each class.
    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
    Returns:
        mean_spectrums: dict of mean spectrum by class
    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        # fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        mean_spectrums[class_names[c]] = mean_spectrum
        plt.show()
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.
    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})
        
def find_n_most_varying(spect_means: dict, n = 4):
    """some txt"""
    means = np.fromiter(spect_means.keys(), dtype=float)
    means.sort()
    selected_values = np.concatenate([means[0:n//2], means[-(n//2):]])
    return list(map(lambda x: spect_means[x], selected_values))

def select_best_spectrums(img:np.array,
                          complete_gt, 
                          target_class_id: int, 
                          n:int = 4):
    """some txt"""
    mask = complete_gt == target_class_id
    class_spectrums = img[mask].reshape(-1, img.shape[-1])

    spect_means = {}
    class_spectrums = np.array([class_spectrums[:, i] for i in 
                                                      range(class_spectrums.shape[1])])
    for i, spectrum in enumerate(class_spectrums[:,:]):
        spect_means[np.mean(spectrum, axis=0)] = i

    best_spectrums = find_n_most_varying(spect_means, n)
    return best_spectrums


def build_dataset(mat: np.ndarray,
                  gt: np.ndarray,
                  selected_bands:list,
                  target_class_id:int,
                  num_samples:int = 3,
                  sample_height:int = 200,
                  sample_width:int = 200,
                  threshold:int = 100,
                  rgb_bands: list = []):
    """Create a list of training samples based on an image, target class and selected spectral bands.
    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        selected_bands: list of number of bands, from which select samples
        target_class: name of target_class, a lot of samples should have this class on it
        labels: list of labels
        num_samples: number of randomly cropped 100x100 images from mat
        threshold: minimum number of pixels of target class
    """
    samples = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    height = mat.shape[0]
    width = mat.shape[1]

    bands_to_exclude = list(set(selected_bands).symmetric_difference(np.arange(0, mat.shape[2])))

    successfuly_generated_samples = 0
    while successfuly_generated_samples < num_samples:
        
        # Ensure the sample size is not larger than the image
        sample_height = min(sample_height, height)
        sample_width = min(sample_width, width)

        # Randomly select the top left pixel of the part you want to sample
        start_x = random.randint(0, width - sample_width)
        start_y = random.randint(0, height - sample_height)

        # Verify that target class is represented at random sample
        cand_gt = deepcopy(gt[start_y:start_y+sample_height, start_x:start_x+sample_width])
        unique, counts = np.unique(cand_gt, return_counts=True)
        class_tags_count = dict(zip(unique, counts))
        if target_class_id in class_tags_count.keys() and class_tags_count[target_class_id] > threshold:
            # Sample the part of the image
            samples_bands = deepcopy(mat[start_y:start_y+sample_height, start_x:start_x+sample_width, :])
            samples_bands = np.asarray(samples_bands)

            samples_bands = np.delete(samples_bands, bands_to_exclude, axis = 2)
            samples_bands = samples_bands.transpose((2, 0, 1))

            sample_gt = deepcopy(gt[start_y:start_y+sample_height, start_x:start_x+sample_width])
            sample_gt[sample_gt != target_class_id] = 0

            mask = sample_gt == target_class_id
            class_spectrums = samples_bands.transpose((1, 2, 0))[mask].reshape(-1, samples_bands.transpose((1, 2, 0)).shape[-1])
            target_brightness = np.mean(class_spectrums, axis=0)
            
            samples.append(Sample(original_img=mat[start_y:start_y+sample_height, 
                                                   start_x:start_x+sample_width, 
                                                   :],
                                  band_img=samples_bands,
                                  labels=sample_gt,
                                  target_brightness=target_brightness))

            successfuly_generated_samples += 1
    
    return samples

def show_results(img:np.ndarray, segmented: Result, gt: np.ndarray, rgb_bands:list, output_file:str):
    """some txt
    Args:
        img: 3D np.array HxWxC representing begin image 
        segmented: Result object
        gt: 2D ground truth
    """
    # Normalize the ground truth mask for coloring
    gt_mask_normalized = gt.astype(float) / gt.max()

    rgb = spectral.get_rgb(img, rgb_bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    best_band_img = deepcopy(segmented.sample[segmented.best_band_id])

    segmented_mask = np.where(best_band_img < 50, 1, 0)
    segmented_mask_normalized = segmented_mask.astype(float) / segmented_mask.max()

    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results', 900, 300)  # Установка нового размера окна
    
    segmented_mask_normalized = np.tile(segmented_mask_normalized, (3, 1, 1)).transpose(1, 2, 0)
    gt_mask_normalized = np.tile(gt_mask_normalized, (3, 1, 1)).transpose(1, 2, 0)

    masked_image = rgb.copy()
    masked_image.resize(segmented_mask_normalized.shape)
    # gt_mask_normalized = gt_mask_normalized.resize(segmented_mask_normalized.shape)

    masked_image_segmented = np.where(segmented_mask_normalized.astype(int),
                             np.array([0,255,0], dtype='uint8'),
                             masked_image)

    masked_image_gt = np.where(gt_mask_normalized.astype(int),
                            np.array([0,0,255], dtype='uint8'),
                            masked_image)

    masked_image_segmented = masked_image_segmented.astype(np.uint8)
    masked_image_gt = masked_image_gt.astype(np.uint8)

    segmented_img = rgb * 0.5 + masked_image_segmented * 0.25 + masked_image_gt * 0.25
    segmented_img = segmented_img.astype(np.uint8)

    combined_image = cv2.hconcat([rgb, segmented_img])

    cv2.imshow('Results', combined_image)

    if not(os.path.exists(output_file)):
        os.mkdir(output_file)
        
    cv2.imwrite(os.path.join(output_file, 'segm_res.png'), combined_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.show()

def evaluate_best_segmentation(segmented: list[dict], 
                               samples: list[Sample], 
                               target_class_id: int,
                               metric:str = "iou"):
    """some txt
    Args:
        segmented: 4D np.array SxCxHxW of segmented pixels of target class within C spectral channels and S samples
        gt: 2D ground truth
    Returns:
        2D array of results by every passed sample. Results are in shape (best_band,
         segmented_img, best_metric_score)
    """
    samples_result = []

    for i, sample in enumerate(segmented):
        band_metrics = dict.fromkeys(sample.keys())
        for band_id, band_segment in sample.items():
            # band_segment have brightness values instead of class label, 
            # so we need to convert them into the mask with class labels
            band_segment_with_labels = deepcopy(band_segment)
            band_segment_with_labels[band_segment_with_labels != 255] = target_class_id
            metric_value = 0
            if metric == "iou":
                metric_value = calculate_iou(samples[i].labels, band_segment_with_labels, target_class_id)
            if metric == "pixelwise":
                metric_value = calculate_pixelwise_accuracy(samples[i].labels, band_segment_with_labels)
            band_metrics[band_id] = metric_value
        
        samples_result.append(Result(best_band_id=max(band_metrics, key=band_metrics.get),
                                     sample=sample,
                                     best_score=max(band_metrics.values())))

    return samples_result

def calculate_pixelwise_accuracy(true: np.array, pred: np.array):
    """calculates pixelwise accuracy metric
    Args:
        true: 2D gt of sample (1 if pixel is target class, 0 otherwise)
        pred: 2D prediction (structure similar to gt)
    """
    correct_pixels = (pred == true).count_nonzero()
    uncorrect_pixels = (pred != true).count_nonzero()
    result = (correct_pixels / (correct_pixels + uncorrect_pixels)).item()

    return result
        
def calculate_iou(true: np.ndarray, pred: np.ndarray, target_class_id: int):
    """calculates iou accuracy metric
    Args:
        true: 2D gt of sample (1 if pixel is target class, 0 otherwise)
        pred: 2D prediction (structure similar to gt)
    """
    # if true.shape != pred.shape:
    #     print("---------------")
    #     return 1.0  # переделать, так как модуль селективного внимания обрезает фотку и нуэно проверять обрезанную уже (в выводе тоде самое)
    #     # raise ValueError("Input arrays must have the same shape")
    
    true_resized = np.resize(true, pred.shape)  # возможно неправильно работает
    # Calculate the intersection and union
    intersection = np.logical_and(true_resized==target_class_id, pred==target_class_id)
    union = np.logical_or(true_resized==target_class_id, pred==target_class_id)
    
    if np.sum(union) == 0:
        iou = 0.0  # Avoid division by zero
    else:
        iou = np.sum(intersection) / np.sum(union)
    
    return iou

def calculate_subimage_from_brightness(brightness_values:np.ndarray, img: np.ndarray):
    """some txt"""
    img[~np.isin(img, brightness_values)] = -1
    return img

def extract_stimuls(img: np.ndarray, stimuls_num:int = 2, method:str = "brightness"):
    """extract stimuls from img based on different methods (brightness, etc.)
    Args:
        img: 2D np.array representing sample img
    Returns:
        list of Stimul object, each of those consists info about position of stimul
        and about some internal values (brightness)
    """
    num_of_chunks = stimuls_num
    if method == "brightness":
        brightness_values = np.sort(np.unique(img.flatten()))

        while (len(brightness_values) % num_of_chunks) != 0:
            brightness_values = brightness_values[:-1]

        stimuls_brightnesses = np.array_split(brightness_values, num_of_chunks)    # значение рандомно поставил, пока ничего лучше не придумал
        
        stimuls = list(map(lambda x: Stimul(img=img, 
                                            sub_img=calculate_subimage_from_brightness(img=deepcopy(img),
                                                                                       brightness_values=x),
                                            stimul_values=x), stimuls_brightnesses))
        return stimuls

def show_contours(img:np.ndarray, contours:list[np.ndarray]):
    """some txt
    Args:
        img: 2D np.array respresenting 1 band of input image
        contours: list of selected contours for this img
    """
    fig, ax = plt.subplots()
    img = np.asarray([img])
    img = img.reshape((img.shape[1], img.shape[2], img.shape[0]))

    img = spectral.get_rgb(img, [0])
    ax.imshow(img)

    if len(contours.shape) == 3:
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    else:
        cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours', 300, 300)  # Установка нового размера окна

        cv2.imshow('Contours', contours)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def find_contour_area(contour:np.ndarray):
    """работает ток для замкнутых контуров, внутри них считает область"""
    return 0.5 * np.abs(np.dot(contour[:, 0], 
                    np.roll(contour[:, 1], 1)) - np.dot(contour[:, 1], np.roll(contour[:, 0], 1)))

def check_default_json_file():
    """создает базовый json файл с гиперпараметрами в директории code/hyperparams"""
    file_name = 'model_baseline.json'
    dir = HYPERPARAMS_DIR

    # Проверка наличия файла
    file_path = os.path.join(dir, file_name)
    if not os.path.exists(file_path):

        with open(file_path, 'w') as f:
            json.dump(DEFAULT_HYPERPARAMS, f, indent=4)

    return file_path

def calculate_contour_points_simple(img):
    """some txt"""
    const1 = 2
    const2 = -0.5
    contour = np.zeros_like(img)
    eps = 1e-6
    change_dir = 0
    grad_x, grad_y, grad_xxx, grad_yyy = compute_derivatives(img)
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            change_dir = 0
            norm_g = np.linalg.norm((grad_x[i, j], grad_y[i, j]))

            print(grad_x[i, j], grad_y[i, j])

            # p_minus_d_x = cv2.Sobel(grad_x - eps, cv2.CV_64F, 1, 0, ksize=3)[i, j]
            # p_plus_d_x = cv2.Sobel(grad_x + eps, cv2.CV_64F, 1, 0, ksize=3)[i, j]
            # p_minus_d_y = cv2.Sobel(grad_y - eps, cv2.CV_64F, 0, 1, ksize=3)[i, j]
            # p_plus_d_y = cv2.Sobel(grad_y + eps, cv2.CV_64F, 0, 1, ksize=3)[i, j]

            grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
            grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)

            # с этим проблема, не работает
            if (grad_xx[max(0, i - 1), j] * grad_xx[min(i, img.shape[0] - 1), j] < 0) or \
               (grad_yy[i, max(0, j - 1)] * grad_yy[i, min(j, img.shape[1] - 1)] < 0):
                change_dir = 1
                print("--------")

            if (grad_xxx[i, j] < const2 and grad_yyy[i, j] < const2) and \
                (norm_g > const1):
                contour[i, j] = 1
                
    return contour

def compute_derivatives(img, kernel_size = 1):
    
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=kernel_size)

    grad_xxx = cv2.Sobel(grad_xx, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_yyy = cv2.Sobel(grad_yy, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    return grad_x, grad_y, grad_xx, grad_yy, grad_xxx, grad_yyy

def extract_cont_simple_sobel(image, k_size):

    blurred_image = cv2.GaussianBlur(image, (k_size, k_size), 0)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = cv2.filter2D(blurred_image, -1, sobel_x)
    grad_y = cv2.filter2D(blurred_image, -1, sobel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    # _, thresholded = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    return np.asarray(magnitude)

def find_most_common_brightness(img):
    """some txt"""
    unique, counts = np.unique(img, return_counts=True)
    brightness_counts = dict(zip(unique, counts))
    most_frequent_brightness = max(brightness_counts, key=brightness_counts.get)
    return most_frequent_brightness

def find_largest_square(image, min_size = 5):
    """some txt"""
    max_square_size = 0
    max_square_coords = [0, 0]

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > 0:
                square_size = 1
                while (i + square_size < len(image) and j + square_size < len(image[0]) and
                       all(image[i + k][j:j + square_size + 1]) for k in range(square_size) and
                       all(image[k][j + square_size] for k in range(square_size))):
                    square_size += 1
                if square_size >= min_size and square_size > max_square_size:
                    max_square_size = square_size
                    max_square_coords = (i, j)

    return max_square_coords, max_square_size


def extract_square_subarray(arr):
    rows, cols = arr.shape
    for size in range(min(rows, cols), 0, -1):
        for i in range(rows - size + 1):
            for j in range(cols - size + 1):
                subarray = arr[i:i+size, j:j+size]
                if np.all(subarray == 1):
                    return [i, j], size

def find_random_square(image: np.ndarray):
    """some txt"""
    row, col = image.shape
    square_size = min(row, col)
    found = False

    while not found and square_size > 0:
        start_row = random.randint(0, row - square_size)
        start_col = random.randint(0, col - square_size)
        subimage = image[start_row:start_row + square_size, start_col:start_col + square_size]

        if np.all(subimage > 0):
            found = True
        else:
            square_size -= 1

    if found:
        return (start_row, start_col), square_size
    else:
        return None, 0
    
def linear_descending_to_0(t):
    """some txt"""
    return max(-0.1*t + 7, 0)

def square_ascending(t):
    """some txt"""
    return min(0.4*t**2, 25)

def exp_dec_with_distance(point_1, point_2):
    """some txt"""
    distance = np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)
    return -np.exp(distance)

def periodically_continued(a, b):
    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)

@periodically_continued(-np.pi, np.pi)
def g(x):
    """some txt"""
    if x < 0.1 and x >= 0:
        return 10*x
    if x < 0.2 and x >= 0.1:
        return -4*x + 1.4
    if x <= np.pi and x >= 0.2:
        return -0.1*x + 0.62
    if x < 0 and x > -np.pi:
        return -(g(-x))

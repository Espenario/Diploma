"""some module docstrings"""
import random
import os
import re
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import spectral
import matplotlib.pyplot as plt
from scipy import io, misc
import imageio
import torch
from source.stimul import Stimul

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

def display_image(img, rgb_bands, other_bands):
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


def build_dataset(mat: np.array,
                  gt,
                  selected_bands:list,
                  target_class_id:int,
                  num_samples:int = 3,
                  threshold:int = 100):
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
    samples_labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    height = mat.shape[0]
    width = mat.shape[1]
    mask = np.nonzero(gt == target_class_id)

    sample_height, sample_width = 100, 100
    successfuly_generated_samples = 0
    while successfuly_generated_samples < num_samples:
        
        # Ensure the sample size is not larger than the image
        sample_height = min(sample_height, height)
        sample_width = min(sample_width, width)

        # Randomly select the top left pixel of the part you want to sample
        start_x = random.randint(0, width - sample_width)
        start_y = random.randint(0, height - sample_height)

        # Verify that target class is represented at random sample
        if len(list(set(np.arange(start_y,start_y+sample_height)) & set(mask[0]))) + \
           len(list(set(np.arange(start_x,start_x+sample_width)) & set(mask[1]))) > threshold:
            # Sample the part of the image
            samples.append(mat[start_y:start_y+sample_height, start_x:start_x+sample_width, :])

            sample_gt = gt[start_y:start_y+sample_height, start_x:start_x+sample_width]
            sample_gt[sample_gt == target_class_id] = 1
            sample_gt[sample_gt != target_class_id] = 0
            samples_labels.append(sample_gt)

            successfuly_generated_samples += 1
    
    samples = np.asarray(samples)

    samples = np.array([samples[:, :, :, i] for i in 
                                      range(samples.shape[3]) if i in selected_bands])
    
    return np.asarray(samples), np.asarray(samples_labels)

def show_results(segmented: np.array, gt: np.array):
    """some txt
    Args:
        segmented: 3D np.array CxHxW of segmented pixels of target class within C spectral channels
        gt: 2D ground truth
    """
    pass

def evaluate_best_segmentation(segmented: list[dict], samples_labels: np.array, metric:str = "iou"):
    """some txt
    Args:
        segmented: 3D np.array CxHxW of segmented pixels of target class within C spectral channels
        gt: 2D ground truth
    Returns:
        2D array of results by every passed sample. Results are in shape (best_band,
         segmented_img, best_metric_score)
    """
    samples_result = []

    for _ in segmented:
        band_metrics = dict.fromkeys(segmented.keys())
        for band_id, band_segment in segmented.items():
            metric = 0
            if metric == 'iou':
                metric = calculate_iou(samples_labels, band_segment)
            if metric == "pixelwise":
                metric = calculate_pixelwise_accuracy(samples_labels, band_segment)
            band_metrics[band_id] = metric
        
        samples_result.append(max(band_metrics, key=band_metrics.get), \
                              segmented[max(band_metrics, key=band_metrics.get)], \
                              max(band_metrics.values()) )

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
        
def calculate_iou(true: np.array, pred: np.array):
    """calculates iou accuracy metric
    Args:
        true: 2D gt of sample (1 if pixel is target class, 0 otherwise)
        pred: 2D prediction (structure similar to gt)
    """
    intersection = np.count_nonzero(np.intersect1d(true, pred))
    union = np.count_nonzero(np.union1d(true, pred))
    result = intersection / union
    return result

def calculate_subimage_from_brightness(brightness_values:np.ndarray, img: np.ndarray):
    """some txt"""
    img[img in brightness_values] = 1
    img[img not in brightness_values] = 0
    return img

def extract_stimuls(img: np.ndarray, method = "brightness"):
    """extract stimuls from img based on different methods (brightness, etc.)
    Args:
        img: 2D np.array representing sample img
    Returns:
        list of Stimul object, each of those consists info about position of stimul
        and about some internal values (brightness)
    """
    if method == "brightness":
        brightness_values = np.sort(np.unique(img.flatten()))
        stimuls_brightnesses = np.split(brightness_values, 5)    # значение рандомно поставил, пока ничего лучше не придумал
        
        stimuls = list(map(lambda x: Stimul(sub_img=calculate_subimage_from_brightness(img=img, brightness_values=x),
                                       stimul_values=x), stimuls_brightnesses))
        return stimuls

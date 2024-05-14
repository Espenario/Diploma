import torch
import torch.utils
import torch.utils.data
import numpy as np
from source.utils import display_image, explore_spectrums, build_dataset, select_best_spectrums


class HyperSpectralTorchData(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperSpectralTorchData, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p-1 and 
                                                                       x < data.shape[0] - p and 
                                                                       y > p-1 and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        """Performs random flip to images in array"""
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        """Add some radiation noise to passed data"""
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        """Add some radiation noise to passed data"""
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
    
class HyperSpectralData():
    """Class for storing hyp data for simple ONN"""

    def __init__(self, data, gt, **params):
        """
        Args:
            data: array of 3D hyperspectral image (may be 1-elem array)
            gt: 2D array of labels
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'un' supervised algorithms
        """
        self.data :np.array = data
        self.gt :np.array = gt
        self.labels_to_ignore = params['ignored_labels']
        self.labels = params['labels']
        self.rgb = params['rbg_bands']
        self.num_bands = params['num_of_bands']
        self.samples: np.array
        self.labels: np.array
        self.selected_bands: np.array
        self.target_class: str
    
    def get_data_info(self):
        print(f'Data image dimensions is {self.data.shape} ')
        print(f'Total number of bands is {self.num_bands}')
        
    def view_data(self):
        """Shows hyperspectral image with 3 sample spectrums and rgb"""
        display_image(self.data[0], self.rgb, np.random.randint(1, self.num_bands, size=3))
        # _ = explore_spectrums(self.data[0], self.gt, self.labels)
    
    def select_chanels(self, method = "expert", n = 4):
        """some txt"""
        if method == "expert":
            self.selected_bands = [55, 41, 12]
        if method == "simple_opt":
            sel_bands = select_best_spectrums(
                img = self.data[0],
                complete_gt = self.gt,
                target_class = self.target_class,
                labels = self.labels,
                n = n
            )
            self.selected_bands = sel_bands
        if method == "advanced_opts":
            self.selected_bands = [[55, 41, 12]]


    def create_samples(self):
        """some txt"""
        self.samples, self.labels = build_dataset(self.data[0],
                                                  self.gt, 
                                                  self.selected_bands,
                                                  self.target_class)
        # print(self.samples[0], labels[0])'

    def specify_target_class(self, target: str):
        """some txt"""
        try:
            target in self.labels
        except Exception as e:
            raise ValueError("Объект для сегментации не представлен в данном датасете") from e
        self.target_class = target

        

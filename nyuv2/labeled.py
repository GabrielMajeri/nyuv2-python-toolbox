import h5py
import numpy as np
from PIL import Image

def rotate_image(image):
    return image.rotate(-90, expand=True)

class LabeledDataset:
    """Python interface for the labeled subset of the NYU dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path):
        """Opens the labeled dataset file at the given path."""
        self.file = h5py.File(path)
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']

    def close(self):
        """Closes the HDF5 file from which the dataset is read."""
        self.file.close()

    def __len__(self):
        return len(self.color_maps)

    def __getitem__(self, idx):
        color_map = self.color_maps[idx]
        color_map = np.moveaxis(color_map, 0, -1)
        color_image = Image.fromarray(color_map, mode='RGB')
        color_image = rotate_image(color_image)

        depth_map = self.depth_maps[idx]
        depth_image = Image.fromarray(depth_map, mode='F')
        depth_image = rotate_image(depth_image)

        return color_image, depth_image

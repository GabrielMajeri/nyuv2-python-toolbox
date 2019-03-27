import h5py
import numpy as np

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
        color_map = np.rot90(self.color_maps[idx], k=-1, axes=(1, 2))
        depth_map = np.rot90(self.depth_maps[idx], k=-1)
        return color_map, depth_map

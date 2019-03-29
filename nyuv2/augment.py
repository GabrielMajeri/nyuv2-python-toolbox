import numpy as np
from PIL import Image

def projection_mask(image):
    """Crops an image to the region where the depth signal is most accurate."""

    return image.crop((40, 44, 601, 471))

class RandomCrop:
    """Randomly crop a portion of the input color/depth maps.

    The size of the crop is fixed when this class is initialized.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, color, depth):
        width, height = depth.size
        new_width, new_height = self.size

        left = np.random.randint(0, width - new_width)
        top = np.random.randint(0, height - new_height)
        right = left + new_width
        bottom = top + new_height

        color = color.crop(box=(left, top, right, bottom))
        depth = depth.crop(box=(left, top, right, bottom))

        return color, depth

class RandomHorizontalFlip:
    """Randomly flips the image and its depth left-to-right."""

    def __call__(self, color, depth):
        flip = np.random.randint(2)
        if flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        return color, depth

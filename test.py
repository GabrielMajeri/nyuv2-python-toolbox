from pathlib import Path
from nyuv2 import LabeledDataset
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = Path('dataset')

def plot_color(ax, color):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title("Color")
    ax.imshow(color)

def plot_depth(ax, depth):
    """Displays a depth map from the NYU dataset."""

    ax.axis('off')
    ax.set_title("Depth")
    ax.imshow(depth, cmap='Spectral')

def plot_color_depth(title, color, depth):
    """Draws the color and depth maps of a scene, side-by-side."""

    fig = plt.figure(title, figsize=(12, 5))

    ax = fig.add_subplot(1, 2, 1)
    plot_color(ax, color)

    ax = fig.add_subplot(1, 2, 2)
    plot_depth(ax, depth)

    plt.show()

def test_labeled_dataset():
    labeled = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')

    plot_color_depth("Labeled Dataset Sample", *labeled[42])

    labeled.close()

test_labeled_dataset()

import numpy as np
from PIL import Image
from .raw.project import depth_rel_to_depth_abs

def color_depth_overlay(color, depth_abs, relative=False):
    """Overlay the depth of a scene over its RGB image to help visualize
    the alignment.

    Requires the color image and the corresponding depth map. Set the relative
    argument to true if the depth map is not already in absolute depth units
    (in meters).

    Returns a new overlay of depth and color.
    """

    assert color.size == depth_abs.size, "Color / depth map size mismatch"

    depth_arr = np.array(depth_abs).astype(np.float32)

    if relative == True:
        depth_arr = depth_rel_to_depth_abs(depth_arr)

    depth_ch = (depth_arr - np.min(depth_arr)) / np.max(depth_arr)
    depth_ch = (depth_ch * 255).astype(np.uint8)
    depth_ch = Image.fromarray(depth_ch)

    r, g, _ = color.split()

    return Image.merge("RGB", (r, depth_ch, g))

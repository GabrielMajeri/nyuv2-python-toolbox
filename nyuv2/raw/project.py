import numpy as np

# Maximum depth of the Kinect's sensor, in meters
MAX_DEPTH = 10.0

def depth_rel_to_depth_abs(depth_rel):
    """Projects a depth image from internal Kinect coordinates to world coordinates.

    The absolute 3D space is defined by a horizontal plane made from the X and Z axes,
    with the Y axis pointing up.

    The final result is in meters."""

    DEPTH_PARAM_1 = 351.3
    DEPTH_PARAM_2 = 1092.5

    depth_abs = DEPTH_PARAM_1 / (DEPTH_PARAM_2 - depth_rel)

    return np.clip(depth_abs, 0, MAX_DEPTH)

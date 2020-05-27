from zipfile import ZipFile
import re

class RawDatasetArchive:
    """Loads a zip file containing (a part of) the raw dataset and
    provides member functions for further data processing.
    """

    def __init__(self, zip_path):
        self.zip = ZipFile(zip_path)
        self.frames = synchronise_frames(self.zip.namelist())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def extract_frame(self, frame, path=None):
        """Extracts a synchronised frame of depth and color images.

        The frame parameter must be a pair of depth and color maps from
        the archive. Optionally the path of an extraction directory can be given.
        """

        return map(lambda name: self.zip.extract(name, path=path), frame)

def synchronise_frames(frame_names):
    """Constructs a list of synchronised depth and RGB frames.

    Returns a list of pairs, where the first is the path of a depth image,
    and the second is the path of a color image.
    """

    # Regular expressions for matching depth and color images
    depth_img_prog = re.compile(r'.+/d-.+\.pgm')
    color_img_prog = re.compile(r'.+/r-.+\.ppm')

    # Applies a regex program to the list of names
    def match_names(prog):
        return map(prog.match, frame_names)

    # Filters out Nones from an iterator
    def filter_none(iter):
        return filter(None.__ne__, iter)

    # Converts regex matches to strings
    def match_to_str(matches):
        return map(lambda match: match.group(0), matches)

    # Retrieves the list of image names matching a certain regex program
    def image_names(prog):
        return list(match_to_str(filter_none(match_names(prog))))

    depth_img_names = image_names(depth_img_prog)
    color_img_names = image_names(color_img_prog)

    # By sorting the image names we ensure images come in chronological order
    depth_img_names.sort()
    color_img_names.sort()

    def name_to_timestamp(name):
        """Extracts the timestamp of a RGB / depth image from its name."""
        _, time, _ = name.split('-')
        return float(time)

    frames = []
    color_count = len(color_img_names)
    color_idx = 0

    for depth_img_name in depth_img_names:
        depth_time = name_to_timestamp(depth_img_name)
        color_time = name_to_timestamp(color_img_names[color_idx])

        diff = abs(depth_time - color_time)

        # Keep going through the color images until we find
        # the one with the closest timestamp
        while color_idx + 1 < color_count:
            color_time = name_to_timestamp(color_img_names[color_idx + 1])

            new_diff = abs(depth_time - color_time)

            # Moving forward would only result in worse timestamps
            if new_diff > diff:
                break

            diff = new_diff

            color_idx = color_idx + 1

        frames.append((depth_img_name, color_img_names[color_idx]))

    return frames

"""This script defines class and methods for image data processing"""

import os
from os.path import exists, splitext
from os.path import join as join_path
import shutil
from utilities import utils as ut
from image_features import features

util = ut.Utils()
img_feat = features.ImageFeatures()

class DataProcess():
    """Reads, processes and prepares image data."""

    def __init__(self):
        """initializes class_labels and defines GLCM
           dissimilarity thresholds to remove outlier
           images.
        """
#         self.data_dir = os.path.abspath(os.path.join('../home_depot_products/'))
        self.duplicate_file_marker = '1'
#         super()


    def flatten_subdir(self, current_dir):
        """Move all files in subdirs to here, then delete subdirs.
           Conflicting files are renamed, with 1 appended to their name."""
        for root, dirs, files in os.walk(current_dir, topdown=False):
            if root != current_dir:
                for name in files:
                    source = join_path(root, name)
                    target = self.handle_duplicates(join_path(current_dir, name))
                    os.rename(source, target)

            for name in dirs:
                os.rmdir(join_path(root, name))

    def handle_duplicates(self, target):
        """Appends a marker to duplicate file names"""
        while exists(target):
            base, ext = splitext(target)
            target = base + self.duplicate_file_marker + ext
        return target

    def read_and_process(self):
        """Reads image files from different class directories
           and filters out the outlier images based on GLCM dissimilarity
           threshold for the given class.
        """
        for cls, _ in util.class_labels.items():
            self.flatten_subdir(util.data_dir + '/' + cls + '/')

    def copy_file(self, src, dest):
        """Copies file from source to destination"""
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)

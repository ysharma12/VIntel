"""Uitility functions and parameters are defined in this script"""

import os
import cv2
from matplotlib import pyplot as plt
import matplotlib.pylab as plb

class Utils:
    """Contains general utility functions to support other classes"""
#     def __init__(self,
#                  data_dir='/Users/yachnasharma/Downloads/' +
#                           'case_study_visual_intelligence/' +
#                           'home_depot_products/'):
    def __init__(self):
        data_dir = os.path.abspath(os.path.join('../data/home_depot_products/'))
        self.data_dir = data_dir
#         self.data_dir = os.path.abspath(os.path.join('../home_depot_products/'))
        self.image_height = 224 # for vgg16
        self.image_width = 224
        self.image_channels = 3
        self.class_labels = {'bar_stool'      : 0, 
                             'bookcase'       : 1,
                             'chandelier'     : 2,
                             'dining_chair'   : 3,
                             'market_umbrella': 4,
                             'night_stands'   : 5,
                             'ottoman'        : 6,
                             'sconces'        : 7,
                             'table_lamp'     : 8,
                             'vases'          : 9
                            }
        self.glcm_thresh = {'bar_stool'      : 44.110820, 
                            'bookcase'       : 42.768328,
                            'chandelier'     : 19.414734,
                            'dining_chair'   : 41.687978,
                            'market_umbrella': 36.852976,
                            'night_stands'   : 54.828513,
                            'ottoman'        : 53.217008,
                            'sconces'        : 25.389607,
                            'table_lamp'     : 31.189404,
                            'vases'          : 19.355556
                            }
        
#         bar_stool          44.110820
# bookcase           42.768328
# chandelier         19.414734
# dining_chair       41.687978
# market_umbrella    36.852976
# night_stands       54.828513
# ottoman            53.217008
# sconces            25.389607
# table_lamp         31.189404
# vases              19.355556

class Plotting(Utils):
    """Contains plotting functions"""
    def __init__(self, figure_size=(16, 6), resize=True, gray=False, normalize=True):
        self.figure_size = figure_size
        self.image_height = 64
        self.image_width = 64
        self.resize = resize
        self.gray = gray
        self.normalize = normalize
    
    def plot_image(self, image_path):
        plt.figure(figsize=self.figure_size)
        image = cv2.imread(image_path)
        if self.resize:
            image = cv2.resize(image, (self.image_width, self.image_height),
                               interpolation=cv2.INTER_CUBIC)
        if self.normalize:
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if self.gray:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plb.imshow(image)
        plb.show()
        
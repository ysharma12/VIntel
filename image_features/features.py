"""This script defines class and methods for feature computation and filtering out images not suitable for training"""

import os, sys
import cv2
import shutil
import numpy as np
from glob import glob

from skimage.feature import greycomatrix, greycoprops
from utilities import utils as ut

util = ut.Utils()
WIDTH = util.image_width
HEIGHT = util.image_height

class ImageFeatures():
    """Computes discrete fourier transform of an image to capture image frequencies.
       The dft image is used to compute the image texture features using GLCM(gray level co-occurence matrices)
    """
    
    def __init__(self):
        """Initializes imagefeatures"""
#         self.image = image
        self.glcm_props = {'dissimilarity': [],
                           'correlation': [],
                           'contrast': [],
                           'homogeneity': [],
                           'asm': [],
                           'energy': [],
                           'quality_labels': [],
                           'class_labels': [],
                           'image_name': []
                           }
#         super()

    def fft_spectrum(self, input_image):
        """Computes Discrete Fourier Transform(DFT) of the input image"""
#         https://docs.opencv.org/2.4.13.7/doc/tutorials/core/
#        discrete_fourier_transform/discrete_fourier_transform.html

        dft = cv2.dft(np.float32(input_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return magnitude_spectrum
    
    def get_glcm_features(self, input_image):
        """Computes glcm features for the give input image"""
        image = np.uint8(input_image)
        glcm = greycomatrix(image, [5], list(np.arange(0, 360, 30)), 256, symmetric=True, normed=True)
        dis = greycoprops(glcm, 'dissimilarity')[0][0]
        cor = greycoprops(glcm, 'correlation')[0][0]
        con = greycoprops(glcm, 'contrast')[0][0]
        hom = greycoprops(glcm, 'homogeneity')[0][0]
        asm = greycoprops(glcm, 'ASM')[0][0]
        energy = greycoprops(glcm, 'energy')[0][0]
        return {'dissimilarity': dis,
            'correlation':cor,
            'contrast':con,
            'homogeneity': hom,
            'asm': asm,
            'energy': energy,
            }


    def update_results(self, result, glcm_props, quality_label, image_name, class_label):
        for k,_ in glcm_props.items():
            if k == 'quality_labels':
                glcm_props[k].append(quality_label)
            elif k == 'image_name':
                glcm_props[k].append(image_name)
            elif k == 'class_labels':
                glcm_props[k].append(class_label)
            else:
                glcm_props[k].append(result[k])
        return glcm_props


    def filter_images(self, move_to_dir=False):
        """Reads images for each class and removes images not suitable for training."""
        
        unread_files = []
        for cls,_ in util.class_labels.items():
            cls_path = util.data_dir + '/' + cls + '/'
            print(cls_path)
            images = glob(os.path.join(cls_path, "*.jpg"))
            for i,img in enumerate(images):
                try:
                    image = cv2.imread(img)
                    image_filename = img.split('/')[-1]
                    height, width, channels = image.shape
#                     print(height, width)
                    if height < 1000 or width < 1000:
                        label = 0
                    else:
                        label = 1
                    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
                    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    magnitude_spectrum = self.fft_spectrum(gray)
                    result = self.get_glcm_features(magnitude_spectrum)
                    self.glcm_props = self.update_results(result, self.glcm_props, label, image_filename, cls)
                    if result['dissimilarity'] > util.glcm_thresh[cls] and move_to_dir:
                        destination_dir = cls_path + '_filtered_out/'
                        # move the image to filtered folder where images not suitable for training are placed
                        if not os.path.exists(destination_dir):
                            os.mkdir(destination_dir)
                        shutil.move(img, destination_dir + image_filename)
                        print('moved:' + img)
                except:
                    print("Could not open image:" + img)
                    unread_files.append(img)

        return self.glcm_props, unread_files

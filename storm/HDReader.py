"""
This module provides the HDReader class for reading HD data.
"""

from utils.HD_load import *
from utils.HD_utils import *
import scanpy as sc


class HDReader:
    """
    A class for reading HD data.

    This class provides methods to read various types of HD data, including images,
    tissue position information, slice features, JSON files, and H5 files.
    The data is stored in class attributes for further processing.
    """

    def read_all(self, image_path, tpl_path, slice_path, sf_path, raw_matrix_path):
        """
        Read all necessary HD data from the specified paths.

        This method calls other reading methods to read the image, tissue position information,
        slice features, JSON file, and H5 file in sequence, and stores the data in class attributes.

        Args:
            image_path (str): The path to the image file.
            tpl_path (str): The path to the tissue position file.
            slice_path (str): The path to the slice feature file.
            sf_path (str): The path to the JSON file containing specific features.
            raw_matrix_path (str): The path to the H5 file containing the raw matrix data.

        Returns:
            None
        """
        self.image = load_image(image_path)
        self.tpl = load_tpl(tpl_path)
        self.metadata = load_slice_feature(slice_path)
        self.sf = load_json(sf_path)
        self.read_h5(raw_matrix_path)

    def read_image(self, image_path):
        """
        Read an image file from the specified path and store it in a class attribute.

        Args:
            image_path (str): The path to the image file.

        Returns:
            None
        """
        self.image = load_image(image_path)

    def read_tpl(self, tpl_path):
        """
        Read tissue position information from the specified path and store it in a class attribute.

        Args:
            tpl_path (str): The path to the tissue position file.

        Returns:
            None
        """
        self.tpl = load_tpl(tpl_path)

    def read_feature(self, slice_path):
        """
        Read slice feature information from the specified path and store it in a class attribute.

        Args:
            slice_path (str): The path to the slice feature file.

        Returns:
            None
        """
        self.metadata = load_slice_feature(slice_path)

    def read_json(self, sf_path):
        """
        Read a JSON file from the specified path and store the data in a class attribute.

        Args:
            sf_path (str): The path to the JSON file.

        Returns:
            None
        """
        self.sf = load_json(sf_path)

    def read_h5(self, raw_matrix_path):
        """
        Read an H5 file containing raw matrix data from the specified path and store it in a class attribute.

        This method uses `scanpy` to read the H5 file and makes the gene names unique.

        Args:
            raw_matrix_path (str): The path to the H5 file containing the raw matrix data.

        Returns:
            None
        """
        self.adata = sc.read_10x_h5(raw_matrix_path)
        self.adata.var_names_make_unique()
     
"""
This module provides the VisiumReader class for reading Visium data.
"""
import os
import json
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from skimage import io
from storm.utils.Visium import *
class VisiumReader:
    """
    A class for reading Visium spatial standard data files.

    This class provides methods to read image files, tissue position information,
    and H5 files from a specified folder, and store relevant data in class attributes.
    """

    def read_all(self, folder_path, method, key, gene_token, medium_token_path=None):
        """
        Read all necessary Visium data files from the specified folder.

        This method calls other reading methods to read the image, tissue position information,
        and H5 file in sequence, and stores the data in class attributes.

        Args:
            folder_path (str): The path to the folder containing the necessary files.
            method (str): The data processing method, such as 'binary', 'raw', 'norm', 'medium', etc.
            key (str): The key used to set the gene token,value should be 'id' or 'symbol'
            gene_token (str): The gene token csv file.
            medium_token_path (str, optional): The path to the medium token file. Defaults to None.

        Returns:
            None

        .. jupyter-execute::
        
            from storm.VisiumReader import VisiumReader
            
            import matplotlib.pyplot as plt
            Reader=VisiumReader()
            Reader.read_all(folder_path="../Visium_Human_Breast_Cancer",gene_token="../gene_token_homologs.csv",method="binary",key="symbol")
            plt.imshow(Reader.raw_he)
            print(Reader.adata , Reader.tissue_position_list.head(5) , Reader.scaleJson)

        """
        self.key = key
        self.method = method
        raw_img_path, raw_tpl_path, json_path, h5_path = get_paths(folder_path)
        self.read_img(raw_img_path)
        self.read_tissue_position(raw_tpl_path, json_path)
        self.read_h5(h5_path, method, key, gene_token, medium_token_path)

    def read_img(self, raw_img_path):
        """
        Read an image file from the specified path and store it in a class attribute.

        If the image has 4 channels (RGBA), it will be converted to 3 channels (RGB).

        Args:
            raw_img_path (str): The path to the raw image file.

        Returns:
            None

        .. jupyter-execute::
        
            from storm.VisiumReader import VisiumReader
            
            import matplotlib.pyplot as plt
            Reader=VisiumReader()
            Reader.read_img("../Visium_Human_Breast_Cancer/spatial/tissue_hires_image.png")
            plt.imshow(Reader.raw_he)

        """
        self.raw_he = io.imread(raw_img_path)
        if self.raw_he.ndim == 3 and self.raw_he.shape[2] == 4:
            self.raw_he = self.raw_he[:, :, :3]

    def read_tissue_position(self, raw_tpl_path, json_path):
        """
        Read tissue position information from CSV and JSON files.

        This method reads the tissue position data from a CSV file, processes the data
        based on the number of columns, sets the index to 'barcode', filters the data,
        converts data types, and reads scale information from a JSON file.

        Args:
            raw_tpl_path (str): The path to the raw tissue position CSV file.
            json_path (str): The path to the JSON file containing scale information.

        Returns:
            None

        .. jupyter-execute::
        
            from storm.VisiumReader import VisiumReader
            Reader=VisiumReader()
            base_path="../Visium_Human_Breast_Cancer/spatial"
            Reader.read_tissue_position(raw_tpl_path=f"{base_path}/tissue_positions_list.csv",
                                        json_path=f"{base_path}/scalefactors_json.json")
            print(Reader.tissue_position_list.head(5))
        
        """
        tpl = pd.read_csv(raw_tpl_path, header=None)
        if len(tpl.columns) == 7:
            self.tissue_position_list = pd.read_csv(raw_tpl_path, header=None, names=['number', 'barcode', 'in_tissue', 'array_row', 'array_col',
                                                                                   'pxl_row_in_fullres', 'pxl_col_in_fullres'])
        else:
            self.tissue_position_list = pd.read_csv(raw_tpl_path, header=None, names=['barcode', 'in_tissue', 'array_row', 'array_col',
                                                                                   'pxl_row_in_fullres', 'pxl_col_in_fullres'])
        self.tissue_position_list.set_index('barcode', inplace=True)
        while str(self.tissue_position_list.iloc[0, 0]) != "0" and str(self.tissue_position_list.iloc[0, 0]) != "1":
            self.tissue_position_list = self.tissue_position_list[1:]
        self.tissue_position_list["in_tissue"] = self.tissue_position_list["in_tissue"].astype('int')
        self.tissue_position_list["pxl_row_in_fullres"] = self.tissue_position_list["pxl_row_in_fullres"].astype('float')
        self.tissue_position_list["pxl_col_in_fullres"] = self.tissue_position_list["pxl_col_in_fullres"].astype('float')
        self.tissue_position_list["array_row"] = self.tissue_position_list["array_row"].astype('float')
        self.tissue_position_list["array_col"] = self.tissue_position_list["array_col"].astype('float')
        self.scaleJson = pd.read_json(json_path, lines=True)
        self.dia = self.scaleJson['spot_diameter_fullres'] * self.scaleJson['tissue_hires_scalef']

    def read_h5(self, h5_path, method, key, gene_token, medium_token_path=None):
        """
        Read H5 or MTX files and process the data based on the specified method.

        This method reads single - cell data from H5 or MTX files, sets the gene token,
        and processes the data matrix according to the specified method.

        Args:
            h5_path (str): The path to the H5 or MTX file.
            method (str): The data processing method, such as 'binary', 'raw', 'norm', 'medium', etc.
            key (Any): The key used to set the gene token.
            gene_token (Any): The gene token.
            medium_token_path (str, optional): The path to the medium token file. Defaults to None.

        Returns:
            None

        .. jupyter-execute::
        
            from storm.VisiumReader import VisiumReader
            Reader=VisiumReader()
            Reader.read_h5(h5_path="../Visium_Human_Breast_Cancer/filtered_feature_bc_matrix.h5",gene_token="../gene_token_homologs.csv",method="binary",key="symbol")
            print(Reader.adata)

        """
        self.method = method
        self.medium_token_path = medium_token_path
        self.key = key
        if h5_path.count(".h5"):
            try:
                adata = sc.read_10x_h5(h5_path)
            except:
                adata = sc.read_h5ad(h5_path)
        elif h5_path.count(".mtx"):
            h5_path = os.path.dirname(h5_path)
            adata = sc.read_10x_mtx(path=h5_path, make_unique=False)
        self.adata = set_gene_token(adata, self.key, gene_token)

        if self.method == "binary":
            self.adata.X = binary_matrix(self.adata.X)
        elif self.method == "raw":
            pass
        # elif type(method) == int:
        #     self.adata.X = np.array([kbins(row, method) for row in self.adata.X])
        elif self.method == "norm":
            sc.pp.normalize_total(self.adata, inplace=True)
            sc.pp.log1p(self.adata)
        elif self.method == "medium":
            with open(self.medium_token_path, "r") as f:
                gene_dict = json.load(f)
            def minMax(col, num):
                try:
                    return col / gene_dict[str(num)]["q50"]
                except:
                    return col
            self.adata.X = np.array([minMax(col, num + 1) for col, num in zip(self.adata.X.T, range(1, 15758))]).T

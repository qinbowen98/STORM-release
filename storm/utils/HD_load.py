import pandas as pd
import numpy as np
import h5py
import json
from skimage import io
def load_image(fulres_path):
    """
    Load an image from the specified path and ensure it is in RGB format.
    
    Args:
    fulres_path (str): The file path to the image.
    
    Returns:
    numpy.ndarray: The loaded image in RGB format.
    """
    fulres_he = io.imread(fulres_path)
    if fulres_he.shape[-1] == 4:
        fulres_he = fulres_he[:, :, :3]
        print("Remove last channel to ensure only RGB!")
    else:
        print("Image is RGB")
    
    return fulres_he

def load_tpl(tpl_path):
    """
    Load a Parquet file and process the template DataFrame by setting the index,
    renaming columns, and selecting specific columns.
    
    Args:
    tpl_path (str): The file path to the Parquet file.
    
    Returns:
    pandas.DataFrame: The processed template DataFrame.
    """
    tpl = pd.read_parquet(tpl_path)
    tpl.set_index('barcode', inplace=True)
    tpl.rename(columns={
        'pxl_row_in_fullres': 'pxl_y_fullres',
        'pxl_col_in_fullres': 'pxl_x_fullres',
        'pxl_row_in_hires': 'pxl_y_hires',
        'pxl_col_in_hires': 'pxl_x_hires',
        'array_row': 'spot_y',
        'array_col': 'spot_x'
    }, inplace=True)
    tpl = tpl[['spot_x', 'spot_y', 'in_tissue']].copy()
    return tpl

def load_slice_feature(slice_path):
    """
    Load metadata and transformation matrix from an HDF5 file.
    
    Args:
    slice_path (str): The file path to the HDF5 file.
    
    Returns:
    dict: The metadata dictionary.
    numpy.ndarray: The transformation matrix from spot_colrow to microscope_colrow.
    """
    with h5py.File(slice_path, 'r') as file:
        metadata_json = file.attrs['metadata_json']
    metadata = json.loads(metadata_json)
    spot_to_microscope = np.array(metadata['transform_matrices']['spot_colrow_to_microscope_colrow'])
    return metadata, spot_to_microscope

def load_json(sf_path):
    """
    Load a JSON file from the specified path.
    
    Args:
    sf_path (str): The file path to the JSON file.
    
    Returns:
    dict: The loaded JSON data as a dictionary.
    """
    with open(sf_path, 'r') as js:
        sf = json.load(js)
    return sf

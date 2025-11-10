from utils.HD_load import *
from utils.HD_utils import *
import gc
import numpy as np

def _to_npy(image_grid):
    """
    Convert specific columns of the image grid DataFrame to NumPy arrays.

    This function extracts the 'acx' and 'acy' columns as 1D NumPy arrays,
    and converts the 'grid' column into a 3D NumPy array where each element
    in the outer array corresponds to a row in the DataFrame, and the inner
    2D arrays contain the first two columns of the 'grid' data.

    Args:
        image_grid (pandas.DataFrame): A DataFrame containing 'acx', 'acy', and 'grid' columns.

    Returns:
        tuple: A tuple containing three NumPy arrays:
               - image_grid_acx: 1D array of 'acx' values.
               - image_grid_acy: 1D array of 'acy' values.
               - image_grid_grid: 3D array of 'grid' values.
    """
    image_grid_acx = image_grid['acx'].values
    image_grid_acy = image_grid['acy'].values
    image_grid_grid = np.array([grid[:, :2] for grid in image_grid['grid'].values])
    return image_grid_acx, image_grid_acy, image_grid_grid

class HDPreprocesser:
    """
    A class for processing High - Definition (HD) data.

    This class provides methods to process the HD data read by an HDReader instance.
    It can process the grid, image, tissue, and matrix data, and save the processed data to files.
    
    """

    def __init__(self, Reader, prefix, token_path):
        """
        Initialize the HDProcesser class.

        Args:
            Reader (HDReader): An instance of the HDReader class that has already read the HD data.
            prefix (str): A prefix string used for various processing steps.
            token_path (str): The path to the file containing gene token information.
        """
        self.prefix = prefix
        self.token_path = token_path
        self.image = Reader.image
        self.tpl = Reader.tpl
        self.metadata, self.spot_to_microscope = Reader.metadata
        self.sf = Reader.sf
        self.adata = Reader.adata

    def process_grid(self, patch_size, dst_res, bin_res=2):
        """
        Generate an image and tissue grid based on the tissue position and spot - to - microscope information.

        Args:
            patch_size (int): The size of the patches used in generating the image grid.
            dst_res (int): The destination resolution for the image grid.
            bin_res (int, optional): The binning resolution. Defaults to 2.

        Returns:
            None
        """
        self.image_grid = generate_image_grid(self.tpl, self.spot_to_microscope, self.prefix, patch_size, dst_res, bin_res)
        self.tissue_grid = generate_tissue_grid(self.image_grid, self.tpl)

    def process_image(self):
        """
        Process the image based on the image grid information.

        This method first converts specific columns of the image grid to NumPy arrays,
        then calculates the pixels using these arrays and the original image,
        and finally post - processes the calculated image.

        Returns:
            None: The processed image is stored in the 'nwimg' attribute of the class.
        """
        image_grid_acx, image_grid_acy, image_grid_grid = _to_npy(self.image_grid)
        nwimg = calculate_pixels(image_grid_acx, image_grid_acy, image_grid_grid, self.image, self.prefix)
        self.nwimg = postp_img(nwimg)

        

    def process_matrix(self):
        """
        Process the matrix data and update the AnnData object.

        This method deletes the image grid, original image, and tissue position information
        to free up memory, then renews the AnnData object based on the tissue grid,
        sets the gene token, and binarizes the matrix data.

        Returns:
            None: The processed AnnData object is stored in the 'adata' attribute of the class.
        """
        del self.image_grid, self.image, self.tpl
        gc.collect()
        self.adata = renew_adata(self.adata, self.tissue_grid)
        tkn_list = pd.read_csv(self.token_path, index_col=0)
        self.adata = set_gene_token(self.adata, tkn_list)
        self.adata.X = binary_matrix(self.adata.X)

    def saveFile(self, imgrid_path=None, fhe_path=None, h5ad_path=None):
        """
        Save the processed data to files.

        This method can save the image grid, processed H&E image, and AnnData object to files
        if the corresponding file paths are provided.

        Args:
            imgrid_path (str, optional): The path to save the image grid as a Parquet file. Defaults to None.
            fhe_path (str, optional): The path to save the processed H&E image. Defaults to None.
            h5ad_path (str, optional): The path to save the AnnData object. Defaults to None.

        Returns:
            None
        """
        if imgrid_path:
            self.image_grid[['cx', 'cy', 'acx', 'acy', 'oldindex', 'in_tissue']].to_parquet(imgrid_path, index=True)
        if fhe_path:
            io.imsave(fhe_path, self.nwimg)
        if h5ad_path:
            self.adata.write(h5ad_path)

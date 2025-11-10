"""
This module provides the VisiumProcesser class for processing Visium spatial data.
It includes methods for processing tissue position data, generating grids, 
manipulating AnnData objects, and saving processed data.
"""
import numpy as np
import pandas as pd
import anndata as ad
from skimage import io
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix, vstack
from skimage.segmentation import expand_labels
from scipy.interpolate import NearestNDInterpolator
from storm.utils.Visium import *


class VisiumPreprocesser:
    """
    A class for processing Visium spatial data.

    This class takes a file reader object and patch size as inputs, 
    and provides a series of methods to process tissue position data, 
    H&E images, and gene expression data. The processed data can be saved to files.
    """

    def __init__(self, Reader, patch_size):
        """
        Initialize the VisiumProcesser class.

        Args:
            Reader (fileReader): An instance of the fileReader class that has already read Visium data.
            patch_size (int): The size of the patches used in grid generation and processing.
        """
        self.patch_size = patch_size
        self.raw_he = Reader.raw_he
        self.tissue_position_list = Reader.tissue_position_list
        self.scaleJson = Reader.scaleJson
        self.dia = Reader.dia
        self.adata = Reader.adata

    def process_tpl(self):
        """
        Process the tissue position data in a CSV - like DataFrame.

        This method modifies the internal DataFrame 'tissue_position_list' directly.
        It sets the index to 'barcode', removes rows until the first barcode starts with '0' or '1',
        converts all columns to float type, scales certain pixel columns, renames columns for clarity,
        and adds two new columns initialized as None.

        Args:
            None (This is an instance method and does not require any arguments.)

        Returns:
            None

        

        """
        while str(self.tissue_position_list.iloc[0, 0]) not in ["0", "1"]:
            self.tissue_position_list = self.tissue_position_list[1:]
        self.tissue_position_list = self.tissue_position_list.astype(float)
        self.tissue_position_list[['pxl_row_in_hires', 'pxl_col_in_hires']] = self.tissue_position_list[
                                                                                ['pxl_row_in_fullres',
                                                                                 'pxl_col_in_fullres']] * \
                                                                             self.scaleJson[
                                                                                 'tissue_hires_scalef'].iloc[0]
        self.tissue_position_list.rename(columns={
            'pxl_row_in_fullres': 'pxl_y_fullres',
            'pxl_col_in_fullres': 'pxl_x_fullres',
            'pxl_row_in_hires': 'pxl_y_hires',
            'pxl_col_in_hires': 'pxl_x_hires',
            'array_row': 'spot_y',
            'array_col': 'spot_x'
        }, inplace=True)
        self.tissue_position_list['top_left_before'] = None
        self.tissue_position_list['top_left_after'] = None

    def round_spot(self):
        """
        Round the coordinates of spots to grid lines.

        This method calculates the radius of a circle based on tissue scale factor and spot diameter.
        It rounds the top - left corner coordinates of each tissue position to the nearest grid line (with 4 as the grid size),
        crops the H&E image using the bounding box of the rounded tissue positions with an additional spot area,
        and finally transforms the tissue position list by normalizing the rounded top - left coordinates.

        Returns:
            None


        """
        circ_radius = 1 * self.scaleJson['tissue_hires_scalef'].iloc[0] * self.scaleJson[
            'spot_diameter_fullres'].iloc[0] * 0.5
        self.circ_radius = circ_radius
        for idx, row in self.tissue_position_list.iterrows():
            top_left_before = (row['pxl_x_hires'] - circ_radius, row['pxl_y_hires'] - circ_radius)
            self.tissue_position_list.at[idx, 'top_left_before'] = top_left_before
            top_left_after_x = round_to_tl(row['pxl_x_hires'] - circ_radius, 4)
            top_left_after_y = round_to_tl(row['pxl_y_hires'] - circ_radius, 4)
            top_left_after = (top_left_after_x, top_left_after_y)
            self.tissue_position_list.at[idx, 'top_left_after'] = top_left_after

        # Cropping H&E via tissue bounding box
        x_values, y_values = zip(*self.tissue_position_list['top_left_after'])
        a_spot = np.ceil(circ_radius * 2).astype(int)
        x_min = max(0, min(x_values))
        y_min = max(0, min(y_values))
        ih, iw, _ = self.raw_he.shape
        self.x_max = min(iw, max(x_values))
        self.y_max = min(ih, max(y_values))
        self.prop_he = self.raw_he[y_min:self.y_max + a_spot, x_min:self.x_max + a_spot]  # add a spot

        # Original tpl
        final_tpl = self.tissue_position_list[['in_tissue', 'spot_y', 'spot_x', 'top_left_after']].copy()
        final_tpl[['tl_x', 'tl_y']] = pd.DataFrame(final_tpl['top_left_after'].tolist(),
                                                  index=final_tpl.index)
        final_tpl.drop(columns=['top_left_after'], inplace=True)
        # normalize coordinate
        self.offset = (min(final_tpl['tl_x']), min(final_tpl['tl_y']))
        final_tpl['tl_x'] -= min(final_tpl['tl_x'])
        final_tpl['tl_y'] -= min(final_tpl['tl_y'])
        self.final_tpl = final_tpl

    def cal_cor(self):
        """
        Generate tissue grid by tissue bounding box.

        This method finds the bounding box of the tissue positions within the tissue.
        It ensures that the dimensions of the bounding box are divisible by the patch size.

        Returns:
            tuple: A tuple containing four integers representing the boundaries of the tissue grid.
                - x_min (int): The minimum x - coordinate of the tissue grid.
                - x_max (int): The maximum x - coordinate of the tissue grid, adjusted to be divisible by the patch size.
                - y_min (int): The minimum y - coordinate of the tissue grid.
                - y_max (int): The maximum y - coordinate of the tissue grid, adjusted to be divisible by the patch size.
         
        """
        filtered_tpf = self.final_tpl[self.final_tpl['in_tissue'] == 1]
        x_max, x_min = min(filtered_tpf['tl_x'].max(), self.x_max), filtered_tpf['tl_x'].min()
        y_max, y_min = min(filtered_tpf['tl_y'].max(), self.y_max), filtered_tpf['tl_y'].min()
        # Ensure dividable by patch size
        x_max = x_min + (((x_max - x_min) + self.patch_size - 1) // self.patch_size) * self.patch_size
        y_max = y_min + (((y_max - y_min) + self.patch_size - 1) // self.patch_size) * self.patch_size
        return x_min, x_max, y_min, y_max

    def generate_grid(self):
        """
        Generate tissue grid.

        This method calculates the tissue grid boundaries using the cal_cor method,
        creates a meshgrid based on the boundaries and grid spacing, 
        and stores the resulting grid in a DataFrame.

        Returns:
            None

        """
        x_min, x_max, y_min, y_max = self.cal_cor()
        grid_spacing = 4
        tl_xs_val = np.arange(x_min, x_max + grid_spacing, grid_spacing)
        tl_ys_val = np.arange(y_min, y_max + grid_spacing, grid_spacing)

        tl_xn, tl_yn = np.meshgrid(tl_xs_val, tl_ys_val)
        tissue = {
            'tl_xn': tl_xn.ravel(),
            'tl_yn': tl_yn.ravel()
        }

        tissue_grid = pd.DataFrame(tissue)
        tissue_grid['index'] = tissue_grid.apply(lambda row: f"s_004_{row['tl_xn']}_{row['tl_yn']}-n", axis=1)

        tissue_grid.set_index('index', inplace=True)
        self.tissue_grid = tissue_grid

    def mark_bc_label(self, row):
        """
        Mark barcode labels in the array_forBarcode.

        This method assigns a number to a specific position in the array_forBarcode based on the center coordinates of a row.

        Args:
            row (pandas.Series): A row from the DataFrame containing 'center_x' and 'center_y' columns.

        Returns:
            None
        """
        self.array_forBarcode[int(row["center_x"] / 4), int(row["center_y"] / 4)] = row.num

    def mark_tissue_label(self, row):
        """
        Mark tissue labels in the array_forIntissue.

        This method attempts to assign the 'in_tissue' value of a row to a specific position in the array_forIntissue.
        If an exception occurs, the method does nothing.

        Args:
            row (pandas.Series): A row from the DataFrame containing 'tl_xn' and 'tl_yn' columns.

        Returns:
            None
        """
        try:
            self.array_forIntissue[int(row["tl_xn"] / 4), int(row["tl_yn"] / 4)] = row["in_tissue"]
        except:
            return

    def mark_tissue_grid(self, row):
        """
        Mark tissue grid labels.

        This method attempts to assign the 'in_tissue' value of a row to a specific index in the tissue_grid DataFrame.
        If the index does not exist, it adds a new row to the DataFrame.

        Args:
            row (pandas.Series): A row from the DataFrame containing 'center_x' and 'center_y' columns.

        Returns:
            None
        """
        index = f"s_004_{int(row['center_x'])}_{int(row['center_y'])}-n"
        try:
            self.tissue_grid.loc[index, "in_tissue"] = row["in_tissue"]
        except:
            self.tissue_grid.loc[index] = [row["center_x"], row["center_y"], row["in_tissue"]]

    def expand_barcode(self):
        """
        Expand barcode information and mark tissue grid labels.

        This method adds center coordinates and a number column to the final_tpl DataFrame,
        initializes two arrays for barcode and tissue labels,
        marks barcode and tissue grid labels,
        finds the nearest in - tissue point for each grid point,
        expands the barcode labels, and creates a DataFrame for barcode information.

        Returns:
            None
        """
        self.final_tpl["center_x"] = self.final_tpl["tl_x"] + 8
        self.final_tpl["center_y"] = self.final_tpl["tl_y"] - 8
        self.final_tpl["num"] = np.arange(1, len(self.final_tpl) + 1)
        self.array_forBarcode = np.zeros(
            shape=(int(self.final_tpl["tl_x"].max() / 4 + 5), int(self.final_tpl["tl_y"].max() / 4 + 5)))
        self.array_forIntissue = np.zeros(
            shape=(int(self.final_tpl["tl_x"].max() / 4 + 5), int(self.final_tpl["tl_y"].max() / 4 + 5)))
        self.final_tpl.apply(self.mark_bc_label, axis=1)
        self.final_tpl.apply(self.mark_tissue_grid, axis=1)
        kd_tree = KDTree(self.final_tpl[['tl_x', 'tl_y']])

        def find_nearest_in_tissue(row):
            if pd.isna(row['in_tissue']):
                _, ind = kd_tree.query([[row['tl_xn'], row['tl_yn']]], k=1)
                nearest_in_tissue = self.final_tpl.iloc[ind[0][0]]['in_tissue']
                return nearest_in_tissue
            else:
                return row['in_tissue']

        self.tissue_grid['in_tissue'] = self.tissue_grid.apply(find_nearest_in_tissue, axis=1)
        self.tissue_grid.apply(self.mark_tissue_label, axis=1)
        self.array_forBarcode = expand_labels(self.array_forBarcode, distance=int(self.dia // 4))
        self.array_forBarcode = self.array_forBarcode * self.array_forIntissue
        barcode_list = [0]
        barcode_list.extend(self.final_tpl.index.tolist())
        pd_barcode = []
        for i, barcode in np.ndenumerate(self.array_forBarcode):
            x = i[0]
            y = i[1]
            pd_barcode.append([f"s_004_{int(x * 4)}_{int(y * 4)}-n", barcode_list[int(barcode)]])
        pd_barcode = pd.DataFrame(pd_barcode, columns=["position", "barcode_55"])
        pd_barcode.index = pd_barcode["position"]
        self.barcode = pd_barcode

    def map_tissue(self):
        """
        Map tissue information to the tissue grid.

        This method expands barcode information, joins the barcode DataFrame to the tissue grid,
        fills missing barcode values, and adds spot coordinates to the tissue grid.

        Returns:
            None



        """
        self.expand_barcode()
        self.tissue_grid = self.tissue_grid.join(self.barcode, how='left', rsuffix='_cor')
        self.tissue_grid['barcode_55'] = self.tissue_grid['barcode_55'].fillna('new')
        self.tissue_grid['spot_x'] = self.tissue_grid['tl_xn'] // 4
        self.tissue_grid['spot_y'] = self.tissue_grid['tl_yn'] // 4

    def find_avg_grid(self):
        """
        Find the average grid and create an AnnData object for it.

        This method filters the tissue grid to include only points with valid barcodes and in - tissue status,
        calculates the average gene expression for each barcode group,
        and creates an AnnData object for the average grid.

        Returns:
            None


        """
        self.avg_grid = self.tissue_grid[
            self.tissue_grid['barcode_55'].isin(self.adata.obs_names) &
            (self.tissue_grid['in_tissue'] == 1)
        ]

        avg_bc = self.avg_grid['barcode_55']
        groups = self.avg_grid.groupby("barcode_55")
        barcode_lis = avg_bc.unique()
        avg_X = np.zeros(shape=self.adata[avg_bc].X.shape)
        cur_length = 0
        obs_index = []
        for barcode in barcode_lis:
            group = groups.get_group(barcode)
            length = len(group)
            avg_X[cur_length:cur_length + length, :] = (self.adata[barcode].X.sum(axis=0) / length) * np.ones(
                shape=(length, 1))
            obs_index.extend(group.index)
            cur_length += length
        avg_X = csr_matrix(avg_X)
        self.avg_adata = ad.AnnData(X=avg_X, var=self.adata.var)
        self.avg_adata.obs.index = obs_index

    def insert_grid(self):
        """
        Interpolate gene expression for the non - average grid points.

        This method filters the tissue grid to include only non - average grid points within the tissue,
        uses nearest neighbor interpolation to estimate gene expression for these points,
        and creates an AnnData object for the interpolated grid.

        Returns:
            None

        """
        self.interp_grid = self.tissue_grid[
            ~self.tissue_grid.index.isin(self.avg_grid.index) &
            (self.tissue_grid['in_tissue'] == 1)
        ].dropna(subset=["tl_xn"], how='any', axis=0)

        orig_coords = self.final_tpl[self.final_tpl["in_tissue"] == 1][["center_x", "center_y"]].values
        interp_coords = self.interp_grid[['tl_xn', 'tl_yn']].values
        orig_expr = self.final_tpl[self.final_tpl["in_tissue"] == 1].index
        interpolator = NearestNDInterpolator(orig_coords, orig_expr)
        interp_expr_before = interpolator(interp_coords)
        wrong_barcode = list(set([x for x in interp_expr_before if x not in self.adata.obs_names.values]))
        print(len([str(x) for x in interp_expr_before if str(x) in self.adata.obs_names.values]))
        interp_expr = self.adata[[str(x) for x in interp_expr_before if str(x) in self.adata.obs_names.values], :].X
        if len(interp_expr) != len(interp_expr_before):
            interp_expr = np.vstack(
                (interp_expr, np.zeros(shape=(len(interp_expr_before) - len(interp_expr), len(self.adata.var_names)))))
        interp_expr = csr_matrix(interp_expr)
        self.interp_adata = ad.AnnData(X=interp_expr, var=self.adata.var)
        self.interp_adata.obs.index = [x for x in self.interp_grid.index.values if x not in wrong_barcode]

    def concat_adata(self):
        """
        Concatenate the average, interpolated, and zero - valued AnnData objects.

        This method creates a zero - valued AnnData object for the remaining grid points,
        concatenates the average, interpolated, and zero - valued AnnData objects,
        and ensures that the indices of the final AnnData object match the tissue grid indices.

        Returns:
            None
        

        """
        zero_grid = self.tissue_grid[
            ~(self.tissue_grid.index.isin(self.avg_grid.index))
            & ~(self.tissue_grid.index.isin(self.interp_grid.index))
        ]
        zero_X = csr_matrix((zero_grid.shape[0], self.adata.shape[1]))
        zero_adata = ad.AnnData(X=zero_X, var=self.adata.var)
        zero_adata.obs.index = zero_grid.index

        fnl_adata_X = vstack([self.avg_adata.X, self.interp_adata.X, zero_adata.X])
        fnl_adata_obs = pd.concat([self.avg_adata.obs, self.interp_adata.obs, zero_adata.obs])
        fnl_adata_var = self.avg_adata.var.copy()
        fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)
        self.fnl_adata = fnl_adata
        assert set(fnl_adata.obs_names) == set(self.tissue_grid.index), "Indices do not match"

    def crop_img(self):
        """
        Crop and pad the H&E image.

        This method calculates the padding needed for the H&E image based on the tissue grid,
        segments the tissue in the image, balances the white color,
        pads the image with white space, and calculates color features before and after processing.

        Returns:
            tuple: A tuple containing six elements:
                - raw_he_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue foreground 
                  in the raw H&E image. Each element corresponds to a color channel (e.g., RGB).
                - raw_bg_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue background 
                  in the raw H&E image. Each element corresponds to a color channel (e.g., RGB).
                - after_he_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue foreground 
                  in the processed H&E image. Each element corresponds to a color channel (e.g., RGB).
                - after_bg_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue background 
                  in the processed H&E image. Each element corresponds to a color channel (e.g., RGB).
                - raw_he (np.ndarray): A 3D NumPy array representing the raw H&E image with shape (height, width, channels).
                - pd_he (np.ndarray): A 3D NumPy array representing the processed and padded H&E image with shape (height, width, channels).
        
            
        """
        tg_xmax, tg_ymax = self.tissue_grid['tl_xn'].max() + 4, self.tissue_grid['tl_yn'].max() + 4

        padding_y, padding_x = max(0, tg_ymax - self.prop_he.shape[0]), max(0, tg_xmax - self.prop_he.shape[1])

        mask = exc_tissue(self.prop_he, method='otsu')

        ih, iw, ic = self.prop_he.shape
        pdh, pdw = ih + padding_y, iw + padding_x
        he_fg, he_bg = self.prop_he[mask], self.prop_he[~mask]
        raw_he_feature = np.median(he_fg, axis=0).astype(int)
        raw_bg_feature = np.median(he_bg, axis=0).astype(int)
        raw_he = self.prop_he
        self.prop_he = white_balance_using_white_point(self.prop_he, ~mask)
        pad_w = np.array([255, 255, 255]).astype(np.uint8)
        pd_he = np.ones((int(pdh), int(pdw), int(ic)), dtype=np.uint8) * pad_w
        pdy1, pdy2 = 0, ih
        pdx1, pdx2 = 0, iw
        pd_he[pdy1:pdy2, pdx1:pdx2] = self.prop_he
        he_fg, he_bg = self.prop_he[mask], self.prop_he[~mask]
        after_he_feature = np.median(he_fg, axis=0).astype(int)
        after_bg_feature = np.median(he_bg, axis=0).astype(int)
        if padding_y > 0:
            pd_he[pdy2:, :] = pad_w
        if padding_x > 0:
            pd_he[:, pdx2:] = pad_w
        self.pd_he = pd_he
        return raw_he_feature, raw_bg_feature, after_he_feature, after_bg_feature, raw_he, self.pd_he

    def process_img(self):
        """
        Process the H&E image.

        This method processes the tissue position data, rounds the spot coordinates,
        generates the tissue grid, maps tissue information, and crops the H&E image.

        Returns:
            tuple: A tuple containing six elements returned by the crop_img method.
                - raw_he_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue foreground 
                  in the raw H&E image. Each element corresponds to a color channel (e.g., RGB).
                - raw_bg_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue background 
                  in the raw H&E image. Each element corresponds to a color channel (e.g., RGB).
                - after_he_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue foreground 
                  in the processed H&E image. Each element corresponds to a color channel (e.g., RGB).
                - after_bg_feature (np.ndarray): A 1D NumPy array representing the median color values of the tissue background 
                  in the processed H&E image. Each element corresponds to a color channel (e.g., RGB).
                - raw_he (np.ndarray): A 3D NumPy array representing the raw H&E image with shape (height, width, channels).
                - pd_he (np.ndarray): A 3D NumPy array representing the processed and padded H&E image with shape (height, width, channels).
        

        """
        self.process_tpl()
        self.round_spot()
        self.generate_grid()
        self.map_tissue()
        return self.crop_img()

    def process_adata(self):
        """
        Process the AnnData object.

        This method finds the average grid, interpolates gene expression for non - average grid points,
        and concatenates the resulting AnnData objects.

        Returns:
            None

        """
        self.find_avg_grid()
        self.insert_grid()
        self.concat_adata()

    def save(self, final_grid_path=None, final_h5ad_path=None, final_png_path=None, final_color_path=None):
        """
        Save the processed data to files.

        This method processes the H&E image, saves color features, the H&E image,
        the tissue grid, and the final AnnData object to files if the corresponding file paths are provided.

        Args:
            final_grid_path (str, optional): The path to save the tissue grid as a CSV file. Defaults to None.
            final_h5ad_path (str, optional): The path to save the final AnnData object. Defaults to None.
            final_png_path (str, optional): The path to save the processed and raw H&E images as PNG files. Defaults to None.
            final_color_path (str, optional): The path to save the color features. Defaults to None.

        Returns:
            None
        """
        raw_he_color, raw_bg_color, after_he_color, after_bg_color, raw_he, _ = self.process_img()
        self.after_bg_color = after_bg_color
        self.tissue_grid = self.tissue_grid[~self.tissue_grid["tl_xn"].isna()]
        if final_color_path:
            with open(final_color_path, "w") as f:
                f.write(str(raw_he_color) + " " + str(raw_bg_color))
            with open(final_color_path.replace("raw_color", "offset"), "w") as f:
                f.write(str(self.offset))
            with open(final_color_path.replace("raw", "after"), "w") as f:
                f.write(str(after_he_color) + " " + str(after_bg_color))
        if final_png_path:
            io.imsave(final_png_path, self.pd_he)
            io.imsave(final_png_path.replace("tissue", "raw_tissue"), raw_he)
        if final_grid_path:
            self.tissue_grid.to_csv(final_grid_path, sep=',', index=True, header=True)
        if final_h5ad_path:
            self.process_adata()
            self.fnl_adata.X = csr_matrix(self.fnl_adata.X)
            self.fnl_adata.write(final_h5ad_path)

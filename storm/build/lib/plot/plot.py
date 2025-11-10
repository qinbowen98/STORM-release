import numpy as np
import pandas as pd

import scanpy as sc
import json
from skimage import io
import matplotlib.pyplot as plt

import math

import h5py
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize

def regen_grid(tissue_position: pd.DataFrame, scale_factor: float = 4,size_factor:float = 40) -> pd.DataFrame:
    """
    Transforms tissue position coordinates using an affine transformation.
    
    Parameters:
        tissue_position (pd.DataFrame): DataFrame containing 'spot_x' and 'spot_y' columns.
        scale_factor (float): Scaling factor for transformation.
    
    Returns:
        pd.DataFrame: Transformed DataFrame with 'pxl_x', 'pxl_y', 'tl_pxl_x', and 'tl_pxl_y' columns.
    """
    x_offset = 1/2 * scale_factor  # Pixel center offset
    #y_offset = (tissue_position['spot_y'].max()+1)* scale_factor - 1/2*scale_factor
    y_offset = size_factor - 1/2*scale_factor
    affine_mtx = np.array([
        [scale_factor, 0, x_offset],
        [0, -scale_factor, y_offset]
    ])
    # print(affine_mtx )
    coords = tissue_position[['spot_x', 'spot_y']].values
    coords_homo = np.hstack([coords, np.ones((coords.shape[0], 1))])  # Homogeneous coordinates
    
    trans_coords = (affine_mtx @ coords_homo.T).T  # Apply affine transformation
    tissue_position[['pxl_x', 'pxl_y']] = trans_coords
    
    scale_length = scale_factor
    tissue_position['tl_pxl_x'] = tissue_position['pxl_x'] - scale_length / 2
    tissue_position['tl_pxl_y'] = tissue_position['pxl_y'] - scale_length / 2
    
    return tissue_position


def _calc_corner_coord(df):
    cx, cy = df['spot_x'].to_numpy(), df['spot_y'].to_numpy() # center point

    dx = np.array([-0.5, -0.5, 0.5, 0.5])
    dy = np.array([-0.5, 0.5, -0.5, 0.5])
    
    corner_cx = cx[:, None] + dx # tl
    corner_cy = cy[:, None] + dy
    
    return corner_cx, corner_cy

def _trans_coord(cx_corners, cy_corners, z=0, spot_to_microscope=None):

    z_array = np.full(cx_corners.shape, z)
    corners_stack = np.stack((cx_corners, cy_corners, z_array), axis=-1)

    transformed = np.dot(corners_stack, spot_to_microscope.T)
    
    px_fr = transformed[..., 0] / transformed[..., 2]
    py_fr = transformed[..., 1] / transformed[..., 2]
    pz = transformed[..., 2]
    return px_fr, py_fr, pz

def process_dataframe(df, spot_to_microscope):
    cx_corners, cy_corners = _calc_corner_coord(df)
    px_fr, py_fr, pz = _trans_coord(cx_corners, cy_corners, z=1, spot_to_microscope=spot_to_microscope)
    df['corner_1_px_fr'], df['corner_1_py_fr'] = px_fr[:, 0], py_fr[:, 0]
    df['corner_2_px_fr'], df['corner_2_py_fr'] = px_fr[:, 1], py_fr[:, 1]
    df['corner_3_px_fr'], df['corner_3_py_fr'] = px_fr[:, 2], py_fr[:, 2]
    df['corner_4_px_fr'], df['corner_4_py_fr'] = px_fr[:, 3], py_fr[:, 3]
    
    return df



def get_patch(img,x_start,x_end,y_start,y_end):
    patch = img[y_start:y_end, x_start:x_end]
    return patch


def extract_patch(HD_grid, HE, HVG_adata, x_hd, y_hd, patch_size, scale_factor):
    # Selecting the patch in HD grid
    # center coord
    patch_HD = HD_grid[(HD_grid["spot_x"] >= x_hd) & (HD_grid["spot_x"] < x_hd + patch_size) & 
                   (HD_grid["spot_y"] >= y_hd) & (HD_grid["spot_y"] < y_hd + patch_size)].copy()

    # Calculating the start and end positions for the HE patch
    x_start = math.ceil(patch_HD["tl_pxl_x"].min()) # 3-18 向右取整数
    x_end = math.ceil(patch_HD["tl_pxl_x"].max() + scale_factor)
    y_start = math.ceil(patch_HD["tl_pxl_y"].min())
    y_end = math.ceil(patch_HD["tl_pxl_y"].max() + scale_factor)

    # Extracting the patch from the HE image
    patch_HE = get_patch(HE, x_start, x_end, y_start, y_end)
    
    # Getting the subset of gene expression data
    ## Even in raw, there still be some sequencing block does not contain any counts?
    adata_subset = HVG_adata[patch_HD.index].copy()

    # Recording the correspondence from HE to HD in in-patch coordinates
    patch_HD['in_patch_pxl_cx'] = patch_HD['pxl_x'] - patch_HD['pxl_x'].min() + 0.5 * (scale_factor - 1)
    patch_HD['in_patch_pxl_cy'] = patch_HD['pxl_y'] - patch_HD['pxl_y'].min() + 0.5 * (scale_factor - 1)
    
    # For the construction of the original LR gene_fg
    # np.filp
    patch_HD['spot_x_in_patch'] = patch_HD['spot_x'] - patch_HD['spot_x'].min()
    patch_HD['spot_y_in_patch'] = patch_HD['spot_y'] - patch_HD['spot_y'].min()
    
    patch_HD.loc[:, 'in_patch_tlx'] = patch_HD['tl_pxl_x'] - patch_HD['tl_pxl_x'].min()#+scale_factor #-0.5  #3-13
    patch_HD.loc[:, 'in_patch_tly'] = patch_HD['tl_pxl_y'] - patch_HD['tl_pxl_y'].min()#+scale_factor#-0.5  #3-13
    
    max_y = patch_HD['spot_y_in_patch'].max()
    patch_HD['spot_y_in_patch'] = max_y - patch_HD['spot_y_in_patch']

    return patch_HE, adata_subset, patch_HD

def plot_patch(ax, patch_HE, adata_subset, patch_HD, scale_factor, gene='Total Counts'):
    # Get gene expression values
    gene_expression = adata_subset.obs_vector(gene)

    # Define color map
    cmap = LinearSegmentedColormap.from_list("custom_red", [(1, 1, 1, 0), (1, 0, 0, 1)], N=8192)
    norm = Normalize(vmin=gene_expression.min(), vmax=gene_expression.max())

    # Plot background image
    ax.imshow(patch_HE)

    # Draw gene expression patches
    for (index, row), expr in zip(patch_HD.iterrows(), gene_expression):
        color = cmap(norm(expr)) if expr > 0 else (1, 1, 1, 0)  # Fully transparent for zero expression
        rect = patches.Rectangle(
            (row['in_patch_tlx'] - 0.5, row['in_patch_tly'] - 0.5), 
            scale_factor, scale_factor,
            linewidth=1, edgecolor='blue', linestyle='--', 
            facecolor=color, #alpha=1 if expr > 0 else 0  # Ensure transparency for zero expression
        )
        ax.add_patch(rect)

    # Set title
    # ax.set_title(f'{gene} localization', fontsize=16, color='r')

    # Add colorbar (specific to this axis)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    # cbar.set_label('Expression Level')
    ax.set_xticks([])  # 移除 X 轴刻度
    ax.set_yticks([])  # 移除 Y 轴刻度
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度标记
    return ax


def plot_patch_with_cropped_HE(ax, HE_img, selected_ROI, adata_subset, gene='Total Counts'):
    """
    Crop H&E image based on the ROI and plot all polygon patches.
    Colored fill indicates gene expression; empty outline for zero expression.

    Parameters:
    - ax: Matplotlib axis where the image and patches will be plotted.
    - HE_img: Full H&E image (numpy array).
    - selected_ROI: DataFrame with polygon corner coordinates (corner_*_px_fr, corner_*_py_fr).
    - adata_subset: AnnData subset aligned with selected_ROI.
    - gene: The gene to visualize.
    """

    # 获取角点坐标列
    x_cols = [col for col in selected_ROI.columns if '_px_fr' in col]
    y_cols = [col for col in selected_ROI.columns if '_py_fr' in col]
    all_x = selected_ROI[x_cols].values.flatten()
    all_y = selected_ROI[y_cols].values.flatten()

    # 计算裁剪区域
    x_min, x_max = int(all_x.min()), int(all_x.max())
    y_min, y_max = int(all_y.min()), int(all_y.max())
    cropped_HE_img = HE_img[y_min:y_max, x_min:x_max]

    # 获取表达数据
    gene_expression = adata_subset.obs_vector(gene)

    # 定义色图
    cmap = LinearSegmentedColormap.from_list("custom_red", [(1, 1, 1, 0), (1, 0, 0, 1)], N=8192)
    norm = Normalize(vmin=gene_expression.min(), vmax=gene_expression.max())

    # 画底图
    ax.imshow(cropped_HE_img)

    # 遍历每个 patch，多边形绘制
    for (index, row), expr in zip(selected_ROI.iterrows(), gene_expression):
        polygon_coords = [
            (row['corner_1_px_fr'] - x_min, row['corner_1_py_fr'] - y_min),
            (row['corner_2_px_fr'] - x_min, row['corner_2_py_fr'] - y_min),
            (row['corner_4_px_fr'] - x_min, row['corner_4_py_fr'] - y_min),
            (row['corner_3_px_fr'] - x_min, row['corner_3_py_fr'] - y_min),
        ]
        if expr > 0:
            color = cmap(norm(expr))
            poly = Polygon(polygon_coords, closed=True, facecolor=color, edgecolor='blue', linewidth=1)
        else:
            # 无表达，仅画边框（浅灰）
            poly = Polygon(polygon_coords, closed=True, facecolor=(1,1,1,0), edgecolor='blue', linewidth=1)
        poly.set_linestyle('dashed')
        ax.add_patch(poly)

    # 标题 & 色条
    # ax.set_title(f'{gene} expression (polygon patch)', fontsize=16, color='r')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    # cbar.set_label('Expression Level')
    ax.set_xticks([])  # 移除 X 轴刻度
    ax.set_yticks([])  # 移除 Y 轴刻度
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度标记

    return ax
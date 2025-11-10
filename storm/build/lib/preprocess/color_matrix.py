"""
This module provides functions for image color processing, 
including stain matrix calculation and color channel extraction.
"""

import spams
import numpy as np
from skimage import io, color, img_as_ubyte, filters
from tqdm import tqdm
from utils.Visium import exc_tissue

def is_uint8_image(x):
    """Check if the input is a uint8 image.

    Args:
        x (np.ndarray): The input array to be checked.

    Returns:
        bool: True if the input is a uint8 image, False otherwise.
    """
    if not is_image(x):
        return False
    if x.dtype != np.uint8:
        return False
    return True


def is_image(x):
    """Check if the input is a valid image array.

    Args:
        x (np.ndarray): The input array to be checked.

    Returns:
        bool: True if the input is a 2D or 3D array, False otherwise.
    """
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim not in [2, 3]:
        return False
    return True


def lum_std(img, percentile=95):
    """Normalize the luminance of an RGB image.

    Args:
        img (np.ndarray): The input RGB image.
        percentile (int, optional): The percentile for luminance normalization. Defaults to 95.

    Returns:
        np.ndarray: The luminance - normalized RGB image.
    """
    img_lab = color.rgb2lab(img)
    light_float = img_lab[:, :, 0].astype(float)
    p = np.percentile(light_float, percentile)
    img_lab[:, :, 0] = np.clip(100 * light_float / p, 0, 100)
    img_std = color.lab2rgb(img_lab)
    img_std = img_as_ubyte(img_std)

    return img_std


def _rgb_to_od(img):
    """Convert an RGB image to optical density (OD) space.

    Args:
        img (np.ndarray): The input RGB image.

    Returns:
        np.ndarray: The image in optical density space.
    """
    mask = (img == 0)
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def _od_to_rgb(od):
    """Convert an optical density (OD) image back to RGB space.

    Args:
        od (np.ndarray): The input OD image.

    Returns:
        np.ndarray: The RGB image.
    """
    #assert od.min() >= 0, "Negative optical density."
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def _norm_mtx_rows(A):
    """Normalize the rows of a matrix.

    Args:
        A (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The matrix with normalized rows.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def _concentration(img, stain_matrix, regularizer=0.01):
    """Calculate the stain concentration in an image.

    Args:
        img (np.ndarray): The input RGB image.
        stain_matrix (np.ndarray): The stain matrix.
        regularizer (float, optional): The regularization parameter. Defaults to 0.01.

    Returns:
        np.ndarray: The stain concentration matrix.
    """
    od = _rgb_to_od(img).reshape((-1, 3))
    # np.dot(od, np.linalg.pinv(stain))
    return spams.lasso(X=od.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T


def macenko_mtx(img, tissue_mask, alpha=99):
    """Calculate the stain matrix using the Macenko method.

    Args:
        img (np.ndarray): The input RGB image.
        tissue_mask (np.ndarray): The tissue mask.
        alpha (float, optional): The percentile for angle selection. Defaults to 99.

    Returns:
        np.ndarray: The calculated stain matrix.
    """
    # Convert RGB to OD
    od = _rgb_to_od(img).reshape((-1, 3))
    od = od[tissue_mask]
    # Eigenvectors of covariance matrix in OD space
    _, V = np.linalg.eigh(np.cov(od, rowvar=False))

    # The two principal eigenvectors
    V = V[:, [2, 1]]

    # Make sure vectors are pointing the right way
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    # Project on this basis
    theta = np.dot(od, V)

    # Angular coordinates with respect to the principal, orthogonal eigenvectors
    phi = np.arctan2(theta[:, 1], theta[:, 0])

    # Min and max angles
    phi_min = np.percentile(phi, 100 - alpha)
    phi_max = np.percentile(phi, alpha)

    # The two principal colors
    v1 = np.dot(V, np.array([np.cos(phi_min), np.sin(phi_min)]))
    v2 = np.dot(V, np.array([np.cos(phi_max), np.sin(phi_max)]))

    # Order of H and E
    if v1[0] > v2[0]:
        stain_mtx = np.array([v1, v2])
    else:
        stain_mtx = np.array([v2, v1])
    # Not normalize here, opt for smart patch
    # stain_mtx = _norm_mtx_rows(stain_mtx)

    return stain_mtx


def vahadane_mtx(img, tissue_mask, regularizer=0.1):
    """Calculate the stain matrix using the Vahadane method.

    Args:
        img (np.ndarray): The input RGB image.
        tissue_mask (np.ndarray): The tissue mask.
        regularizer (float, optional): The regularization parameter. Defaults to 0.1.

    Returns:
        np.ndarray: The calculated stain matrix.
    """
    assert is_uint8_image(img), "Image should be RGB uint8."
    # convert to OD and ignore background
    od = _rgb_to_od(img).reshape((-1, 3))
    od = od[tissue_mask]

    # do the dictionary learning
    stain_mtx = spams.trainDL(X=od.T, K=2, lambda1=regularizer, mode=2,
                                   modeD=0, posAlpha=True, posD=True, verbose=False).T
    # order H and E.
    # H on first row.
    if stain_mtx[0, 0] < stain_mtx[1, 0]:
        stain_mtx = stain_mtx[[1, 0], :]
        
    # Not normalize here, opt for smart patch
    # stain_mtx = _norm_mtx_rows(stain_mtx)
    return stain_mtx


def exc_he(img, stain_matrix):
    """Extract Hematoxylin and Eosin channels from an image.

    Args:
        img (np.ndarray): The input RGB image.
        stain_matrix (np.ndarray): The stain matrix.

    Returns:
        tuple: A tuple containing the Hematoxylin and Eosin channel images.
    
    Examples:
        >>> from preprocess.color_matrix import exc_he
        >>> from skimage import io
        >>> import os
        >>> # Use relative path to load the image
        >>> img_path = 'Visium_Human_Breast_Cancer/spatial/tissue_hires_image.png'
        >>> # stain matrix can be calculated from the image by Cal_matrix
        >>> # Create a dummy stain matrix for demonstration
        >>> stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
        >>> h_image, e_image = exc_he(img, stain_matrix)
    """
    # Calculate the concentration matrix
    src_conc = _concentration(img, stain_matrix)

    # Extract Hematoxylin and Eosin images
    h_image = _od_to_rgb(np.outer(src_conc[:, 0], stain_matrix[0]).reshape(img.shape))
    e_image = _od_to_rgb(np.outer(src_conc[:, 1], stain_matrix[1]).reshape(img.shape))

    return h_image, e_image


def process_img(he, patch_size=1000, tissue_threshold=0.4, max_patches=20, method="vahadane"):
    """Process an image to calculate the stain matrix.

    Args:
        he (np.ndarray): The input HE - stained image.
        patch_size (int, optional): The size of image patches. Defaults to 1000.
        tissue_threshold (float, optional): The tissue threshold. Defaults to 0.4.
        max_patches (int, optional): The maximum number of patches. Defaults to 20.
        method (str, optional): The method for stain matrix calculation. Defaults to "vahadane".

    Returns:
        np.ndarray: The calculated and normalized stain matrix.
    """
    height, width, _ = he.shape
    img_size = max(height, width)

    if img_size < 5000:
        img_std = lum_std(he, percentile=95)
        mask = exc_tissue(img_std, method='otsu').astype(bool)
        if method == "vahadane":
            mtx = vahadane_mtx(img_std, mask.reshape((-1,)), regularizer=0.1)
        elif method == "macenko":
            mtx = macenko_mtx(img_std, mask.reshape((-1,)))
        else:
            raise ValueError("method error")
        mtx = _norm_mtx_rows(mtx)
        return mtx
    else:
        mtx_list = []
        count = 0    
        
        with tqdm(total=max_patches, desc="Processing patches") as pbar:
            while count < max_patches:
                y = np.random.randint(0, height - patch_size + 1)
                x = np.random.randint(0, width - patch_size + 1)
                patch = he[y:y+patch_size, x:x+patch_size]
                img_std = lum_std(patch, percentile=95)
                mask = exc_tissue(img_std, method='otsu').astype(bool)
                if np.mean(mask) < tissue_threshold:
                    continue
                
                if method == "vahadane":
                    mtx = vahadane_mtx(img_std, mask.reshape((-1,)), regularizer=0.1)
                elif method == "macenko":
                    mtx = macenko_mtx(img_std, mask.reshape((-1,)))
                else:
                    raise ValueError("method error")
                mtx_list.append(mtx)
                count += 1
                pbar.update(1)
        mtx_array = np.array(mtx_list)
        median_matrix = np.median(mtx_array, axis=0)
        normalized_median_matrix = _norm_mtx_rows(median_matrix)
        return normalized_median_matrix


class Cal_CMatrix:
    """A class for calculating color matrices."""

    def readFile(self, raw_img_path):
        """Read an image file and store it in the class instance.

        Args:
            raw_img_path (str): The path to the raw image file.
        """
        he = io.imread(raw_img_path)

        if he.ndim == 3 and he.shape[2] == 4:
            he = he[:, :, :3]
        self.he = he

    def get_cmtx(self, method="vahadane"):
        """Get the calculated color matrix.

        Args:
            method (str, optional): The method for stain matrix calculation. should be "vahadane" or "macenko".

        Returns:
            np.ndarray: The calculated stain color matrix.
        
        """

        return process_img(self.he, patch_size=1000, tissue_threshold=0.45, max_patches=20, method=method)
import pandas as pd
import time
import numpy as np
from numba import njit, prange
from skimage import exposure
import anndata as ad
from scipy.sparse import  hstack,vstack, csr_matrix,issparse

@njit(parallel=True)
def _meshgrid(nw_ax, nw_ay):
    gx = np.zeros((len(nw_ax), len(nw_ay)), dtype=np.int64)
    gy = np.zeros((len(nw_ax), len(nw_ay)), dtype=np.int64)
    for i in prange(len(nw_ax)): # pylint: disable=not-an-iterable
        for j in range(len(nw_ay)):
            gx[i, j] = nw_ax[i]
            gy[i, j] = nw_ay[j]
    return gx.ravel(), gy.ravel()

# Generate grid coordinate
@njit
def _grid_pts(min_x, max_x, min_y, max_y, stp, sf, patch_size):
    # Divideable by patch size
    # NOT normalize here? should be normalized at pixel
    # This is center corrdinate, no need to add one
    max_x = min_x + (((max_x - min_x) + patch_size - 1) // patch_size) * patch_size
    max_y = min_y + (((max_y - min_y) + patch_size - 1) // patch_size) * patch_size
    nw_ax = (np.arange(min_x, max_x, stp) * sf).astype(np.int64)
    nw_ay = (np.arange(min_y, max_y, stp) * sf).astype(np.int64)
    return _meshgrid(nw_ax, nw_ay)

# Generate image pixel coordinate in original spot coordinate
@njit
def _ac_coord(fx, fy, idxsf):
    return (fx / idxsf).astype(np.int64), (fy / idxsf).astype(np.int64)

# Calculate coordinate transformation
@njit
def _trans_coord(cx, cy, z, spot_to_microscope):
    transformed = np.dot(np.vstack((cx, cy, z)).T, spot_to_microscope.T)
    px_fr = transformed[:, 0] / transformed[:, 2]
    py_fr = transformed[:, 1] / transformed[:, 2]
    pz = transformed[:, 2]
    return px_fr, py_fr, pz

# Calculate image pixel HD correspondancy
@njit(parallel=True)
def _corn_coord(cx, cy, z, spot_to_microscope, dst_res):
    offsets = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float64)
    num_offsets = offsets.shape[0]
    num_points = len(cx)
    
    grid_list = np.empty((num_points, num_offsets, 3), dtype=np.float64)

    for i in prange(num_points): # pylint: disable=not-an-iterable
        base_x = cx[i]
        base_y = cy[i]
        base_z = z[i]
        for j in range(num_offsets):
            dx, dy = offsets[j]
            cx_corner = base_x + dx * dst_res
            cy_corner = base_y + dy * dst_res
            # Precompute transformation to avoid multiple dot calls
            transformed_corner = np.dot(spot_to_microscope, np.array([cx_corner, cy_corner, base_z]))
            inv_pz = 1.0 / transformed_corner[2]
            px_fr_corner = transformed_corner[0] * inv_pz
            py_fr_corner = transformed_corner[1] * inv_pz
            pz_corner = transformed_corner[2]
            grid_list[i, j] = (px_fr_corner, py_fr_corner, pz_corner)
    return grid_list
# ==========================================
# Build a lookup table for in_tissue and barcode mapping
# TODO: change here, make look up table do not rely on tpl soley
@njit(parallel=True)
def _bd_lkt(tpl_acx, tpl_acy, tpl_in_tissue, tpl_index, grid_size, max_acx, max_acy):
    lookup_table = -np.ones(((max_acx + 1) * grid_size, (max_acy + 1) * grid_size, 2), dtype=np.int64)  # [index, in_tissue]
    for j in prange(len(tpl_acx)): # pylint: disable=not-an-iterable
        x = tpl_acx[j]
        y = tpl_acy[j]
        for dx in range(grid_size):
            for dy in range(grid_size):
                x_offset = x * grid_size + dx
                y_offset = y * grid_size + dy
                #print(x,y,x_offset,y_offset,tpl_index[j])
                if x_offset < lookup_table.shape[0] and y_offset < lookup_table.shape[1]:
                    lookup_table[x_offset, y_offset, 0] = tpl_index[j]
                    lookup_table[x_offset, y_offset, 1] = tpl_in_tissue[j]
    return lookup_table

# Mapping in_tissue and barcode via lookup table
@njit(parallel=True)
def _mp_idx_it(acx, acy, lookup_table, grid_size):
    oldindex_list = np.zeros(len(acx), dtype=np.int64)
    in_tissue_list = np.zeros(len(acx), dtype=np.int64)
    for i in prange(len(acx)): # pylint: disable=not-an-iterable
        x = acx[i] #* grid_size
        y = acy[i] #* grid_size
        if x < lookup_table.shape[0] and y < lookup_table.shape[1] and lookup_table[x, y, 0] != -1:
            oldindex_list[i] = lookup_table[x, y, 0]
            in_tissue_list[i] = lookup_table[x, y, 1]
        else:
            oldindex_list[i] = -1
            in_tissue_list[i] = 0
    return oldindex_list, in_tissue_list

# Mapping none barcode
def _mp_null_bc(index):
    if isinstance(index, str) and index.startswith('i_002um_') and index.endswith('-n'):
        parts = index.split('_')
        if len(parts) >= 4:
            num1 = int(parts[2])
            num2 = int(parts[3].split('-')[0])
            new_num1 = num1 // 4
            new_num2 = num2 // 4
            formatted_oldindex = f"s_002um_{new_num2:05d}_{new_num1:05d}-1" # Ensure consistency not -2
            return formatted_oldindex
    return None


@njit
def angle_from_centroid(vertex, centroid_x, centroid_y):
    return np.arctan2(vertex[1] - centroid_y, vertex[0] - centroid_x)

# Rearrange polygon vertex: clockwise / anti-clockwise
@njit
def _vtx(vertices):
    centroid_x = 0.0
    centroid_y = 0.0
    for vertex in vertices:
        centroid_x += vertex[0]
        centroid_y += vertex[1]
    
    centroid_x /= len(vertices)
    centroid_y /= len(vertices)
    
    angles = np.empty(len(vertices))
    for i, vertex in enumerate(vertices):
        angles[i] = angle_from_centroid(vertex, centroid_x, centroid_y)
    
    indices = np.argsort(angles)
    #sorted_vertices = np.empty_like(vertices)
    sorted_vertices = np.empty(vertices.shape)
    for i, idx in enumerate(indices):
        sorted_vertices[i] = vertices[idx]
    
    return sorted_vertices

# Gauss's Area Calculation
@njit
def _shoelace(poly):
    poly = _vtx(poly)
    n = len(poly)
    area = 0.0
    for i in range(n - 1):
        area += poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]
    area += poly[-1][0] * poly[0][1] - poly[0][0] * poly[-1][1]
    return 0.5 * abs(area)

# Calculate intersection point of 2 lines
@njit
def _ints_pts(p1, p2, p3, p4):
    s1_x = p2[0] - p1[0]
    s1_y = p2[1] - p1[1]
    s2_x = p4[0] - p3[0]
    s2_y = p4[1] - p3[1]

    denominator = (-s2_x * s1_y + s1_x * s2_y)
    if denominator == 0:
        return None  # Parallel

    s = (-s1_y * (p1[0] - p3[0]) + s1_x * (p1[1] - p3[1])) / denominator
    t = (s2_x * (p1[1] - p3[1]) - s2_y * (p1[0] - p3[0])) / denominator

    if 0 <= s <= 1 and 0 <= t <= 1:
        i_x = p1[0] + (t * s1_x)
        i_y = p1[1] + (t * s1_y)
        return np.array([i_x, i_y])
    return None
    
# Decide whether a point is in a polygon
@njit
def _pts_in_poly(point, poly):
    x, y = point
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Pixel grid intersection
@njit
def _pxl_ints(poly, pixel):
    (px, py) = pixel
    pixel_edges = [
        [(px, py), (px + 1, py)],
        [(px + 1, py), (px + 1, py + 1)],
        [(px + 1, py + 1), (px, py + 1)],
        [(px, py + 1), (px, py)]
    ]
    intersections = np.zeros((len(poly) * len(pixel_edges), 2))
    count = 0
    for i in range(len(poly)):
        for edge in pixel_edges:
            inter = _ints_pts(poly[i], poly[(i + 1) % len(poly)], edge[0], edge[1])
            if inter is not None:
                found = False
                for k in range(count):
                    if np.all(intersections[k] == inter):
                        found = True
                        break
                if not found:
                    intersections[count] = inter
                    count += 1
    return intersections[:count]

# Calculate intersection area per pixel
@njit
def intersection_area(poly, pixel):
    (px, py) = pixel
    
    intersections = np.zeros((8, 2), dtype=np.float64)
    count = 0

    # Check pixel in polygon
    ints = _pxl_ints(poly, pixel)
    for i in range(len(ints)):
        intersections[count] = ints[i]
        count += 1
    # Check polygon vertex
    for vertex in poly:
        if px <= vertex[0] < px + 1 and py <= vertex[1] < py + 1:
            intersections[count] = vertex
            count += 1
    # Check pixel point
    pixel_corners = np.array([(px, py), (px + 1, py), (px + 1, py + 1), (px, py + 1)])
    for corner in pixel_corners:
        if _pts_in_poly(corner, poly):
            intersections[count] = corner
            count += 1
    if count < 3:
        return 0
    # Only meaningful
    return _shoelace(intersections[:count])

# Calculate a polygon's AABB with integer coordinate
@njit
def _poly_aabb(polygon_coords):
    min_x = np.min(polygon_coords[:, 0])
    max_x = np.max(polygon_coords[:, 0])
    min_y = np.min(polygon_coords[:, 1])
    max_y = np.max(polygon_coords[:, 1])

    # Rounding
    min_x_int = int(np.floor(min_x))
    max_x_int = int(np.ceil(max_x))
    min_y_int = int(np.floor(min_y))
    max_y_int = int(np.ceil(max_y))
    
    # Pixel coord
    return (min_x_int, min_y_int, max_x_int, max_y_int)

# Calculate the overall intersection area as weight
@njit
def _ins_w(polygon_coords, xmin, ymin, xmax, ymax):
    region_size = (xmax - xmin, ymax - ymin)  # (width, height)
    intersection_areas = np.zeros((region_size[1], region_size[0]))
    for i in range(region_size[1]):
        for j in range(region_size[0]): 
            pixel = (j + xmin, i + ymin)  # (x, y) format
            area = intersection_area(polygon_coords, pixel)
            intersection_areas[i, j] = area
    return intersection_areas

# Compatiable with numba
@njit
def _swa(arr, axis):
    if axis == (0, 1):
        result = np.zeros((3,), dtype=np.float64)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result += arr[i, j]
        return result
    else:
        raise ValueError("Unsupported axis")

# Real do polygon filter
@njit
def cwpxl(img, grid, pad_w):
    ih, iw, ic = img.shape
    xmin, ymin, xmax, ymax = _poly_aabb(grid)

    aw = _ins_w(grid, xmin, ymin, xmax, ymax)
    aw = aw[:, :, np.newaxis]
    y1 = max(0, ymin)
    y2 = min(ih, ymax) if ymax > 0 else 0

    x1 = max(0, xmin)
    x2 = min(iw, xmax) if xmax > 0 else 0

    cpy1 = max(0, y1 - ymin)
    cpy2 = cpy1 + (y2 - y1)
    cpx1 = max(0, x1 - xmin)
    cpx2 = cpx1 + (x2 - x1)
    
    cpw = xmax - xmin 
    cph = ymax - ymin 

    # NOT 255! 
    cpimg = np.ones((cph, cpw, ic), dtype=np.uint8) * pad_w
    if y1 < y2 and x1 < x2 and cpy1 < cph and cpx1 < cpw:
        cpimg[cpy1:cpy2, cpx1:cpx2] = img[y1:y2, x1:x2]
    if ymin < 0:
        cpimg[:cpy1, :] = pad_w 
    if xmin < 0:
        cpimg[:, :cpx1] = pad_w 
    if ymax > ih:
        cpimg[cpy2:, :] = pad_w
    if xmax > iw:
        cpimg[:, cpx2:] = pad_w 
    
    ws = _swa(cpimg * aw[:cpimg.shape[0], :cpimg.shape[1], :], (0, 1))
    tw = np.sum(aw[:cpimg.shape[0], :cpimg.shape[1]])
    avg_pxl = (ws / tw).reshape((3,)).astype(np.float64)
    return avg_pxl



# Texture mapping a pixel
@njit
def calculate_pixel(img, grid, pad_w):
    avg_pxl = cwpxl(img, grid, pad_w)
    return avg_pxl

# Decide out-of-tissue padding value
# mask = np.all(fulres_he > 200, axis=-1)
# del mask
def _bg_pxl(fulres_he):
    pad_w = np.median(fulres_he[np.all(fulres_he > 200, axis=-1)], axis=0).astype(np.uint8)
    # print("The background pixel value imputed is: ", pad_w)
    return pad_w

# Parallel calculation
@njit(parallel=True)
def _calculate_pixels_chunk(image_grid_acx_chunk, image_grid_acy_chunk, image_grid_grid_chunk, pad_w, img, nwimg):
    for i in prange(image_grid_acx_chunk.shape[0]): # pylint: disable=not-an-iterable
        acx = image_grid_acx_chunk[i]
        acy = image_grid_acy_chunk[i]
        grid = image_grid_grid_chunk[i]
        avg_pxl = calculate_pixel(img, grid, pad_w)
        nwimg[int(acy), int(acx), :] = avg_pxl

def calculate_pixels(image_grid_acx, image_grid_acy, image_grid_grid, img, prefix, chunk_size=5000):  
    h = np.max(image_grid_acy) + 1
    w = np.max(image_grid_acx) + 1
    pad_w = _bg_pxl(img)
    nwimg = np.zeros((h, w, 3), dtype=np.float64)
    num_chunks = (image_grid_acx.shape[0] + chunk_size - 1) // chunk_size

    start_time = time.time()
    
    for chunk in range(num_chunks):
        start = chunk * chunk_size
        end = min(start + chunk_size, image_grid_acx.shape[0])
        _calculate_pixels_chunk(image_grid_acx[start:end], image_grid_acy[start:end], image_grid_grid[start:end], pad_w, img, nwimg)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Pixel generation of {prefix} complete. Total execution time: {elapsed_time:.2f} seconds.")
    
    return nwimg

# Final image is range from [0,1]
# Convert it to [0,255]
def postp_img(img):
    tmp = np.flipud(img)
    tmp = exposure.rescale_intensity(tmp, in_range='image', out_range=(0, 1))
    # https://scikit-image.org/docs/stable/api/skimage.html
    # tmp = img_as_ubyte(tmp)
    return tmp

def renew_adata(adata, tissue_grid):
    """
    Update the AnnData object with tissue grid information, ensuring consistency
    between the indices of the tissue grid and the AnnData observations.

    Args:
    raw_bcfmtx_path (str): The file path to the 10X HDF5 file.
    tissue_grid (pandas.DataFrame): The tissue grid DataFrame.

    Returns:
    ad.AnnData: The updated AnnData object.
    """
    

    # -1/-2 consistency
    nu_bcidx = set(tissue_grid.index) - set(adata.obs_names)

    if nu_bcidx:
        nu_bcidx = list(nu_bcidx)
        nu_obs = pd.DataFrame(index=nu_bcidx)
        # nu_X = np.zeros((len(nu_bcidx), adata.n_vars),dtype=np.float16)
        nu_X = csr_matrix((len(nu_bcidx), adata.n_vars), dtype=np.float16)
        adata_nu = ad.AnnData(
            X=nu_X,
            obs=nu_obs,
            var=adata.var
        )
        fnl_X = vstack([adata.X, adata_nu.X])
        fnl_obs = pd.concat([adata.obs, adata_nu.obs])
        fnl_var = adata.var.copy()
        
        adata = ad.AnnData(X=fnl_X, obs=fnl_obs, var=fnl_var)
        # print("New observations added.")
    else:
        print("No new observations to add.")
    
    return adata

def binary_matrix(matrix):
    matrix[matrix!= 0] = 1
    return matrix.astype(int)

def _to_mtx(adata_subset, patch_tissue):
    x_pos = patch_tissue['sx_ip'].astype(int).values
    y_pos = patch_tissue['sy_ip'].astype(int).values
        
    height, width = y_pos.max() + 1, x_pos.max() + 1
    n_genes = adata_subset.shape[1]
    mtx = np.zeros((height, width, n_genes))
    
    if issparse(adata_subset.X):
        adata_x = adata_subset.X.toarray()
    else:
        adata_x = adata_subset.X    
        
    mtx[y_pos, x_pos, :] = adata_x
    sparse_mtx = csr_matrix(mtx.reshape(-1, n_genes))
    # sparse_mtx.toarray().reshape(height, width, n_genes)
    # np.array_equal(mtx, restored_mtx)
    return sparse_mtx

def set_gene_token(adata, tkn_list):
    gene_names = adata.var_names

    human_matches = tkn_list['HGNC_symbol'].isin(gene_names).sum()
    mouse_matches = tkn_list['MGI_symbol'].isin(gene_names).sum()

    if human_matches > mouse_matches:
        gene_token = set(tkn_list['ENSG_ID'])
        tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
        tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
    elif mouse_matches > human_matches:
        gene_token = set(tkn_list['ENSMUSG_ID'])
        tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
        tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
    else:
        raise ValueError("Cannot determine the species from gene names")
    
    # Extract ID
    adata.var['gene_ids'] = adata.var['gene_ids'].str.split('.', expand=True)[0]

    # Drop .
    intkn_adata = adata[:, adata.var['gene_ids'].isin(gene_token)].copy()
    intkn_adata.var_names = intkn_adata.var['gene_ids'].values
    intkn_adata.var = intkn_adata.var.iloc[:, 0:0]

    intkn_genes = set(intkn_adata.var_names)
    miss_gene = [gene for gene in gene_token if gene not in intkn_genes]

    # var should be a df
    miss_gene = pd.DataFrame(miss_gene, columns=['gene_ids']).set_index('gene_ids')

    miss_X = csr_matrix((adata.n_obs, len(miss_gene)))
    miss_adata = ad.AnnData(X=miss_X, var=miss_gene)
    miss_adata.obs.index = adata.obs.index

    fnl_adata_X = hstack([intkn_adata.X, miss_adata.X])
    fnl_adata_obs = intkn_adata.obs.copy()
    fnl_adata_var = pd.concat([intkn_adata.var, miss_adata.var])
    fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)

    token_order = [fnl_adata.var_names.get_loc(name) for name in tkn_info['gene_ids'].values]
    fnl_adata.var = fnl_adata.var.iloc[token_order]

    if issparse(adata.X):
        fnl_adata.X = fnl_adata.X[:, token_order]
    else:
        fnl_adata = fnl_adata[:, token_order]

    fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info.reset_index(), left_on='index', right_on='gene_ids', how='left')
    fnl_adata.var.set_index('symbol', inplace=True)
    fnl_adata.var = fnl_adata.var.drop(['index'], axis=1)

    return fnl_adata

def generate_image_grid(tpl, spot_to_microscope, prefix, patch_size=224, dst_res=0.5, bin_res=2):
    stp = dst_res / bin_res
    min_x, max_x = tpl["spot_x"].min(), tpl["spot_x"].max()
    min_y, max_y = tpl["spot_y"].min(), tpl["spot_y"].max()  # here are the center of spot coordinate
    sf = 10 ** np.ceil(np.log2(bin_res / dst_res))
    idxsf = (dst_res / bin_res) * 10 ** np.ceil(np.log2(bin_res / dst_res))
    grid_size = int(bin_res / dst_res)
    
    start_time = time.time()

    # Calculate grid points
    spot_patchsize = int(patch_size/grid_size)
    fx, fy = _grid_pts(min_x, max_x, min_y, max_y, stp, sf, spot_patchsize)
    acx, acy = _ac_coord(fx, fy, idxsf)
    # This is actually image pixel not HD sequencing grid!
    index = ['i_002um_' + str(x) + '_' + str(y) + '-n' for x, y in zip(acx, acy)]
    
    # Create tpl data for mapping
    acx_tpl = (tpl["spot_x"].values - min_x).astype(np.int64)
    acy_tpl = (tpl["spot_y"].values - min_y).astype(np.int64)
    tpl_in_tissue = tpl["in_tissue"].to_numpy().astype(np.int64)
    tpl_index = np.arange(len(tpl)).astype(np.int64)
    
    max_acx = acx_tpl.max().astype(np.int64)
    max_acy = acy_tpl.max().astype(np.int64)
    
    # Build lookup table
    # print("Building lookup table...")
    lookup_table = _bd_lkt(acx_tpl, acy_tpl, tpl_in_tissue, tpl_index, grid_size, max_acx, max_acy)
    # Compute oldindex and in_tissue
    # print("Computing oldindex and in_tissue...")
    oldindex_list, in_tissue_list = _mp_idx_it(acx, acy, lookup_table, grid_size)
    
    # Compute px_fr, py_fr, pz
    # print("Computing transformed coordinates...")
    cx = fx / sf
    cy = fy / sf
    z = np.ones(len(fx), dtype=int)
    px_fr, py_fr, pz = _trans_coord(cx, cy, z, spot_to_microscope)
    
    # Compute corner coordinates and store in grid_list
    # print("Computing corner coordinates...")
    grid_list =  _corn_coord(cx, cy, z, spot_to_microscope, dst_res)
    
    # Map oldindex back to original tpl index
    oldindex_mapped = np.array([tpl.index[i] if i != -1 else None for i in oldindex_list])
    # Create a DataFrame from the computed lists
    image_grid= pd.DataFrame({
        'cx': cx,
        'cy': cy,
        'acx': acx,
        'acy': acy,
        'index': index,
        'z': z,
        'oldindex': oldindex_mapped,
        'in_tissue': in_tissue_list,
        'px_fr': px_fr,
        'py_fr': py_fr,
        'pz': pz,
        'grid': list(grid_list)
    })

    image_grid.set_index('index', inplace=True)
    # Map none barcode, these are newly generated unit
    if image_grid['oldindex'].isnull().any():
        # print("New barcode added to ensure patchify!")
        image_grid.loc[image_grid['oldindex'].isnull(), 'oldindex'] = image_grid[image_grid['oldindex'].isnull()].apply(lambda row: _mp_null_bc(row.name), axis=1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Grid generation of {prefix} complete. Total execution time: {elapsed_time:.2f} seconds.")

    
    return image_grid

def generate_tissue_grid(image_grid, tpl):
    """
    Generate a tissue grid by merging a template DataFrame with an image barcode,
    adding missing indices, and calculating new columns.

    Args:
    image_grid (pandas.DataFrame): The DataFrame containing the image grid.
    tpl (pandas.DataFrame): The template DataFrame.

    Returns:
    pandas.DataFrame: The generated tissue grid.
    """
    tissue_grid = tpl.copy()
    nw_bcidx = list(set(image_grid['oldindex'].tolist()) - set(tissue_grid.index))
    nw_rw = []
    for idx in nw_bcidx:
        parts = idx.split('_')
        spot_x = int(parts[3].split('-')[0])
        spot_y = int(parts[2])
        in_tissue = 0
        nw_rw.append({'spot_x': spot_x, 'spot_y': spot_y, 'in_tissue': in_tissue})
    new_tpl = pd.DataFrame(nw_rw, index=nw_bcidx)
    tissue_grid = pd.concat([tissue_grid, new_tpl])
    tissue_grid['tl_xn'] = tissue_grid['spot_x'] * 4
    tissue_grid['tl_yn'] = tissue_grid['spot_y'] * 4

    return tissue_grid
from skimage import color, filters
import numpy as np
import os
import math
import pandas as pd
import anndata as ad
from scipy.sparse import issparse, csr_matrix,hstack
def exc_tissue(img, method='lumin', l_threshold=0.8):
    """
    Generate a tissue mask from an RGB image using either luminance thresholding or Otsu's method.

    This function takes an RGB image and creates a binary mask to identify the tissue region within the image.
    It supports two methods: luminance thresholding and Otsu's method for automatic threshold selection.

    Args:
        img (ndarray): An RGB image represented as a NumPy array with shape (height, width, 3).
        method (str, optional): The method to use for generating the tissue mask. 
            Options are 'lumin' for luminance thresholding and 'otsu' for Otsu's method. Defaults to 'lumin'.
        l_threshold (float, optional): The luminance threshold value used when `method` is set to 'lumin'. 
            Values range from 0 to 1. Defaults to 0.8.

    Returns:
        ndarray: A binary mask representing the detected tissue region in the input image. 
            It has the same height and width as the input image.

    Raises:
        ValueError: If the `method` argument is not 'lumin' or 'otsu'.
    """
    if method not in ['lumin', 'otsu']:
        raise ValueError("Method should be 'lumin' or 'otsu'")
    
    if method == 'lumin':
        img_lab = color.rgb2lab(img)
        light_float = img_lab[:, :, 0] / 100.0  # L* channel range is [0, 100], normalize to [0, 1]
        mask = light_float < l_threshold
    elif method == 'otsu':
        gray_img = color.rgb2gray(img)
        otsu_threshold = filters.threshold_otsu(gray_img)
        mask = gray_img < otsu_threshold
    
    # Check it's not empty
    # assert mask.sum() == 0, "Empty tissue mask computed"
    
    return mask


def white_balance_using_white_point(img, mask):
    """
    Perform white balance on an RGB image using the white point method.

    This function adjusts the color balance of an RGB image by calculating the white point 
    from the masked region of the image. It then applies gain factors to each color channel 
    to achieve a more natural color representation.

    Args:
        img (ndarray): An RGB image represented as a NumPy array with shape (height, width, 3).
            The pixel values should be in the range [0, 255].
        mask (ndarray): A binary mask with the same height and width as the input image.
            The mask is used to select the region of interest for calculating the white point.

    Returns:
        ndarray: The white - balanced RGB image with the same shape as the input image.
            The pixel values are in the range [0, 255].
    """
    img_float = img.astype(np.float32) / 255.0
    img_mask=img*(np.expand_dims(mask,axis=2)*np.ones(shape=(1,1,3)))
    wr, wg, wb = np.percentile(img_mask[:,:,0],95)/255,np.percentile(img_mask[:,:,1],95)/255,np.percentile(img_mask[:,:,2],95)/255
    if(not wr):
        wr=1
    if(not wg):
        wg=1
    if(not wb):
        wb=1
    gain_r = 1.0 / wr
    gain_g = 1.0 / wg
    gain_b = 1.0 / wb
    balanced_img = img_float.copy()
    balanced_img[:, :, 0] *= gain_r
    balanced_img[:, :, 1] *= gain_g
    balanced_img[:, :, 2] *= gain_b
    balanced_img = np.clip(balanced_img, 0, 1)
    balanced_img = (balanced_img * 255).astype(np.uint8)
    return balanced_img

def split_gene_id(gene_id):
    return gene_id.split("_")[-1].split(".")[0]


def set_gene_token(adata,key,token_path):
    tkn_list=pd.read_csv(token_path)
    gene_names =  pd.Series(adata.var_names).str.replace("__","_").apply(split_gene_id).values
    if(key=="id"):
        human_matches = tkn_list['ENSG_ID'].isin(gene_names).sum()
        mouse_matches = tkn_list['ENSMUSG_ID'].isin(gene_names).sum()
    elif(key=="symbol"):
        human_matches = tkn_list['HGNC_symbol'].isin(gene_names).sum()
        mouse_matches = tkn_list['MGI_symbol'].isin(gene_names).sum()
    else:
        raise ValueError("Key are supposed to be id or symbol")
    print(f"gene id matches:\nhuman:{human_matches} \nmouse:{mouse_matches}")
    if (key == "id"):
        if human_matches > mouse_matches:
            gene_token = set(tkn_list['ENSG_ID'])
            tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
            tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
            adata_1 = adata_token(adata,gene_token,tkn_info,key)
            if(mouse_matches > 10000):
                gene_token = tkn_list['MGI_symbol']
                tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
                tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
                adata_2=adata_token(adata,gene_token,tkn_info,key)
                return ad.AnnData(X=adata_1.X+adata_2.X,obs=adata_1.obs,var=adata_1.var)
            else:
                return adata_1
        elif(mouse_matches>human_matches):
            gene_token = set(tkn_list['ENSMUSG_ID'])
            tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
            tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
            adata_1=adata_token(adata,gene_token,tkn_info,key)
            if(human_matches>10000):
                gene_token = set(tkn_list['ENSG_ID'])
                tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
                tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
                adata_2 = adata_token(adata,gene_token,tkn_info,key)
                return ad.AnnData(X=adata_1.X+adata_2.X,obs=adata_1.obs,var=adata_1.var)
            return adata_1
        else:
            raise ValueError("Cannot determine the species from gene names")
    if(key=="symbol"):
        if human_matches>mouse_matches:
            gene_token = set(tkn_list['HGNC_symbol'])
            tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
            tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
            adata_1= adata_token(adata,gene_token,tkn_info,key)
            if(mouse_matches>10000):
                gene_token = tkn_list['MGI_symbol']
                tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
                tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
                adata_2=adata_token(adata,gene_token,tkn_info,key)
                return ad.AnnData(X=adata_1.X+adata_2.X,obs=adata_1.obs,var=adata_1.var)
            return adata_1
        elif(mouse_matches>human_matches):
            gene_token = tkn_list['MGI_symbol']
            tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
            tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
            adata_1= adata_token(adata,gene_token,tkn_info,key)
            if(human_matches>10000):
                gene_token = set(tkn_list['HGNC_symbol'])
                tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
                tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
                adata_2=adata_token(adata,gene_token,tkn_info,key)
                return ad.AnnData(X=adata_1.X+adata_2.X,obs=adata_1.obs,var=adata_1.var)
            return adata_1
        else:
            raise ValueError("Cannot determine the species from gene names")


def adata_token(adata,gene_token,tkn_info,key):
    # Extract ID
    if(key=='id'):
        adata.var['gene_ids'] = adata.var['gene_ids'].apply(split_gene_id).values
        adata.var.index=adata.var["gene_ids"].values
        col="gene_ids"
    else:
        adata.var["index"] = adata.var.index
        adata.var["symbol"] = adata.var["index"].apply(split_gene_id).values
        adata.var_names = pd.Series(adata.var["symbol"]).values
        col="symbol"
    adata.var_names_make_unique()
    # Drop .
    intkn_adata = adata[:, adata.var.index.isin(gene_token)].copy()
    intkn_genes = set(intkn_adata.var_names)
    miss_gene = [gene for gene in gene_token if gene not in intkn_genes]

    # var should be a df
    miss_gene = pd.DataFrame(miss_gene, columns=[col]).set_index(col)
    miss_X = csr_matrix((adata.n_obs, len(miss_gene)))
    miss_adata = ad.AnnData(X=miss_X, var=miss_gene)
    miss_adata.obs.index = adata.obs.index
    fnl_adata_X = hstack([intkn_adata.X, miss_adata.X])
    fnl_adata_obs = intkn_adata.obs.copy()
    fnl_adata_var = pd.concat([intkn_adata.var, miss_adata.var])
    fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)
    
    token_order = [fnl_adata.var_names.get_loc(name) for name in tkn_info[col].values]
    fnl_adata.var = fnl_adata.var.iloc[token_order]
    if issparse(fnl_adata.X):
        fnl_adata.X = fnl_adata.X.toarray()[:, token_order]
    else:
        fnl_adata = fnl_adata[:, token_order]
    try:
        fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info, left_on='index', right_on=col, how='left')
        fnl_adata.var = fnl_adata.var.drop(['index'], axis=1)
    except:
        fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info, left_on=col, right_on=col, how='left')
    fnl_adata.var.set_index(fnl_adata.var.dropna(how='any',axis=1).columns[0], inplace=True)
    return fnl_adata

def round_to_tl(value,grid_spacing):
    return math.floor(value / grid_spacing) * grid_spacing

def binary_matrix(matrix):
    matrix[matrix!= 0] = 1
    return matrix.astype(int)
def get_paths(folder_path):
    '''
    Get detail path for img , json , csv and h5(mtx)

    Args:
    folder_path:the sample folder path for sample
        example:
            path
            ├──spatial
            │    ├──GSM5026146_S4_tissue_hires_image.png
            │    ├──GSM5026146_scalefactors_json.json
            │    └──GSM5026146_tissue_positions_list.csv
            └──filtered_feature_bc_matrix.h5
    Returns:
    raw_img_path:path to deal img
    raw_tpl_path:path to read tissue position
    json_path:path to read scale factors JSON file
    h5_path:path to read 
    '''
    folder_path=folder_path.replace("\n","")
    raw_img_path , raw_tpl_path , json_path , h5_path="","","",""
    for file in os.listdir(folder_path):
        file_path=os.path.join(folder_path,file)
        if(file_path.lower().count("raw") and (file_path.count("h5") or file_path.count(".mtx")) and (not file_path.count("total"))):
            h5_path=file_path
            break  
        elif((file_path.count("h5") or file_path.count(".mtx")) and (not file_path.count("total"))):
            h5_path=file_path
        
    for file in os.listdir(os.path.join(folder_path,"spatial")):
        file_path=os.path.join(folder_path+"/spatial/",file)
        file=file.lower()
        if(file.count("position") and file.count(".csv")):
            raw_tpl_path=file_path
        if(file.count("scalefactor") and file.count(".json")):
            json_path=file_path
        if(file_path.lower().count("hires")):
            raw_img_path=file_path
    if (not (raw_img_path and raw_tpl_path and json_path and h5_path)):
        raise ValueError("some files miss")
    return raw_img_path,raw_tpl_path,json_path,h5_path
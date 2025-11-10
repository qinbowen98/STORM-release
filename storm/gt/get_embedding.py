import yaml
import torch
from skimage import io
from ..models.Storm import Storm
from ..pp.VisiumPreprocesser import VisiumPreprocesser
from ..VisiumReader import VisiumReader
import os
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from numba import njit, prange
import gc


# Reset dict key as class attribute
class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def get_encoder(config_path, ckpt_path, device='cpu'):
    
    """Load and initialize the STORM encoder model.

    Loads the configuration file, constructs the model architecture,
    loads pre-trained weights, and sets the model to evaluation mode.

    Args:
        config_path (str): Path to the model configuration YAML file
        ckpt_path (str): Path to the model checkpoint file (.pth)
        device (str, optional): Computing device to run the model on (e.g., 'cpu' or 'cuda')

    Returns:
        torch.nn.Module: Initialized STORM model ready for inference
    """

    # load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # to attribute
    model_config = AttrDict(config['model'])

    # build model
    model = Storm(model_config)

    # load ckpt
    state_dict = torch.load(ckpt_path, map_location="cpu")["base_model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    # set eval mode & to GPU
    model.eval()
    model.to(device)
    return model

def get_embedding_RGB(config_path, ckpt_path,rgb,res,device="cpu"):
    
    """Extract feature embedding vectors from RGB images.

    Args:
        config_path (str): Path to the model configuration file
        ckpt_path (str): Path to the model checkpoint file (.pth)
        rgb (torch.Tensor): RGB image data with shape (batchsize, 224, 224, 3)
        res (torch.Tensor): Resolution parameter
        device (str, optional): Computing device to run the model on

    Returns:
        torch.Tensor: Embedding vectors extracted from RGB images
    """

    model = get_encoder(config_path, ckpt_path, device=device)
    res = res.to(device).unsqueeze(0).unsqueeze(0)
    rgb = rgb.to(device).permute(0, 3, 1, 2)
    all_embedding = model.forward_rgb(rgb,res)
    return all_embedding

def get_embedding_all(config_path, ckpt_path,rgb,expr,res,device="cpu"):
    
    """Extract joint embedding vectors from RGB images and expression matrices.

    Args:
        config_path (str): Path to the model configuration file
        ckpt_path (str): Path to the model checkpoint file (.pth)
        rgb (torch.Tensor): RGB image data with shape (batchsize, 224, 224, 3)
        expr (torch.Tensor): Gene expression matrix with shape (batchsize, 56, 56, gene_number)
        res (torch.Tensor): Resolution parameter
        device (str, optional): Computing device to run the model on

    Returns:
        torch.Tensor: Joint embedding vectors from RGB images and expression matrices
    """

    model = get_encoder(config_path, ckpt_path, device=device)
    res = res.to(device).unsqueeze(0).unsqueeze(0)
    rgb = rgb.to(device).permute(0, 3, 1, 2)
    expr = expr.to(device).permute(0, 3, 1, 2)

    all_embedding = model.forward_all(rgb,expr,res)
    return all_embedding


def get_expr(config_path, ckpt_path,rgb,res,device="cpu"):

    """Predict gene expression matrix from RGB images.

    Args:
        config_path (str): Path to the model configuration file
        ckpt_path (str): Path to the model checkpoint file (.pth)
        rgb (torch.Tensor): RGB image data with shape (batchsize, 224, 224, 3)
        res (torch.Tensor): Resolution parameter
        device (str, optional): Computing device to run the model on

    Returns:
        torch.Tensor: Predicted gene expression matrix
    """

    model = get_encoder(config_path, ckpt_path, device=device)
    res = res.to(device).unsqueeze(0).unsqueeze(0)
    rgb = rgb.to(device).permute(0, 3, 1, 2)
    expr = model.forward_rgb_to_expr(rgb, res)
    return expr


def get_rgb_and_expr(tif_path,expr_path):
    
    """Load TIF images and corresponding expression matrices.

    Args:
        tif_path (str): Path to the TIF image file
        expr_path (str): Path to the gene expression matrix file (.pt format)

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing RGB image,
            expression matrix, and resolution parameter
    """

    img = io.imread(tif_path)
    rgb = torch.from_numpy(img).float().unsqueeze(0)
    expr = torch.load(expr_path, weights_only=True)#.unsqueeze(0)
    res=torch.full((1,), 0.5, dtype=torch.float32).unsqueeze(0)
    return rgb,expr,res

@njit(parallel=True)
def process_patch_mask(x_coords, y_coords, x_st, y_st, patch_size):
    """使用numba加速掩码计算"""
    mask = np.zeros(len(x_coords), dtype=np.bool_)
    for i in prange(len(x_coords)):
        if (x_coords[i] >= x_st and x_coords[i] < x_st + patch_size and
                y_coords[i] >= y_st and y_coords[i] < y_st + patch_size):
            mask[i] = True
    return mask

class IntegratedDataset(Dataset):
    def __init__(self, processed_patches, patch_coordinates: str,
                 resolution: float = 4.0, target_size=(14, 14)):
        self.patches = processed_patches
        self.coordinates = patch_coordinates
        self.resolution = resolution
        self.target_size = target_size

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        try:
            he_patch, expr_matrix = self.patches[idx]
            x, y = self.coordinates[idx]

            # 图像预处理
            if not isinstance(he_patch, np.ndarray):
                raise ValueError(f"HE patch is not a numpy array: {type(he_patch)}")

            if len(he_patch.shape) != 3:
                raise ValueError(f"HE patch has wrong dimensions: {he_patch.shape}")

            if he_patch.max() > 2:
                he_patch = he_patch / 255.0
            rgb_tensor = torch.from_numpy(he_patch).float()

            # 处理表达式矩阵
            if isinstance(expr_matrix, csr_matrix):
                expr = torch.from_numpy(expr_matrix.toarray()).float()
            else:
                expr = torch.from_numpy(expr_matrix).float()

            size = int(math.sqrt(expr_matrix.shape[0]))
            expr = expr.reshape(size, size, -1)

            data = {
                "coords": torch.tensor([x, y], dtype=torch.int),
                "expr": expr,
                "rgb": rgb_tensor
            }

            res = torch.full((1,), self.resolution, dtype=torch.float32)
            return data,  res

        except Exception as e:
            raise e

def process_patches(x_st, y_st, patch_size, processer):
    """处理单个patch"""
    # 检查边界条件
    if x_st + patch_size > processer.pd_he.shape[1] or y_st + patch_size > processer.pd_he.shape[0]:
        return None

    # 提取patch
    patch_he = processer.pd_he[y_st:y_st + patch_size, x_st:x_st + patch_size].copy()

    # 检查patch尺寸
    if patch_he.shape[:2] != (patch_size, patch_size):
        raise ValueError(f"Invalid patch shape: {patch_he.shape}")

    # 计算mask
    x_coords = processer.tissue_grid['tl_xn'].values
    y_coords = processer.tissue_grid['tl_yn'].values
    tissue_mask = process_patch_mask(x_coords, y_coords, x_st, y_st, patch_size)

    patch_tissue = processer.tissue_grid[tissue_mask]

    if len(patch_tissue) > 0:
        patch_indices = patch_tissue.index
        patch_mtx = processer.fnl_adata[patch_indices].X
        return (patch_he, patch_mtx), (x_st, y_st)

    return None

def collect_patches(processer, patch_size, stride, grid_bounds, batch_size=256):
    """分批收集patches以控制内存使用"""
    grid_xmin, grid_xmax, grid_ymin, grid_ymax = grid_bounds
    processed_patches = []
    patch_coordinates = []
    start_time = time.time()

    total_patches = ((grid_xmax - grid_xmin - patch_size) // stride + 1) * \
                    ((grid_ymax - grid_ymin - patch_size) // stride + 1)

    with tqdm(total=total_patches, desc="Collecting patches", position=0) as pbar:
        for x_st in range(grid_xmin, grid_xmax - patch_size + 1, stride):
            for y_st in range(grid_ymin, grid_ymax - patch_size + 1, stride):
                result = process_patches(x_st, y_st, patch_size, processer)
                if result is not None:
                    patch_data, coords = result
                    he_patch, expr_matrix = patch_data

                    # 检查 patch 形状
                    if he_patch.shape[:2] != (patch_size, patch_size):
                        continue

                    processed_patches.append(patch_data)
                    patch_coordinates.append(coords)

                    # 当收集到足够的patches时，返回当前 batch
                    if len(processed_patches) >= batch_size:
                        elapsed = time.time() - start_time
                        yield processed_patches, patch_coordinates
                        processed_patches = []
                        patch_coordinates = []

                pbar.update(1)

    # 返回最后一批数据
    if processed_patches:
        yield processed_patches, patch_coordinates


@torch.no_grad()
def process_and_infer_optimized_rgb2emb(processer, model, patch_size=224, stride=16,
                                batch_size=32,
                                target_size=(14, 14), device='cuda'):
    """优化版本：直接将推理结果放入大图对应位置"""
    start_time = time.time()
    patch_count = 0
    batch_count = 0

    # 计算网格边界
    grid_bounds = (
        int(processer.tissue_grid['tl_xn'].min()),
        int(processer.tissue_grid['tl_xn'].max()),
        int(processer.tissue_grid['tl_yn'].min()),
        int(processer.tissue_grid['tl_yn'].max())
    )

    # 计算大图尺寸
    feature_size = target_size[0]  # 假设target_size是正方形
    max_x = grid_bounds[1]
    max_y = grid_bounds[3]
    # target_height = int(max_y / stride + feature_size)
    # target_width = int(max_x / stride + feature_size)

    # 使用更安全的尺寸计算方式
    target_height = int(np.ceil(max_y / stride)) + feature_size
    target_width = int(np.ceil(max_x / stride)) + feature_size

    # 初始化大图和计数器
    large_expr_embedding = torch.zeros((target_height, target_width, 1024),
                                       dtype=torch.float32, device=device)
    large_rgb_embedding = torch.zeros((target_height, target_width, 1024),
                                      dtype=torch.float32, device=device)
    count_image = torch.zeros((target_height, target_width, 1),
                              dtype=torch.float32, device=device)

    # 分批处理patches
    for processed_patches, patch_coordinates in collect_patches(
            processer, patch_size, stride, grid_bounds, batch_size=256
    ):
        if not processed_patches:
            continue

        dataset = IntegratedDataset(
            processed_patches=processed_patches,
            patch_coordinates=patch_coordinates,
            target_size=target_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        for batch_data in dataloader:
            data_batch,res = batch_data
            coords = data_batch["coords"]

            # 准备数据并推理
            res = res.unsqueeze(-1).unsqueeze(-1).to(device)
            rgb = data_batch["rgb"].to(device, non_blocking=True).permute(0, 3, 1, 2)
            all_embedding = model.forward_rgb(rgb,res)
            rgb_emb = all_embedding.view(-1, *target_size, 1024)
            # ⚠️ 关键修复5：添加边界检查的特征放置
            for i in range(len(coords)):
                x, y = coords[i]
                start_x = int(x / stride)
                start_y = int(y / stride)
                end_x = start_x + feature_size
                end_y = start_y + feature_size

                # 边界检查
                if end_x > target_width or end_y > target_height:
                    # 调整到边界内
                    end_x = min(end_x, target_width)
                    end_y = min(end_y, target_height)
                    
                    # 如果调整后区域太小，跳过
                    if end_x <= start_x or end_y <= start_y:
                        continue
                    
                    # 调整特征块大小
                    actual_height = end_y - start_y
                    actual_width = end_x - start_x
                    rgb_patch = rgb_emb[i][:actual_height, :actual_width, :]
                else:
                    rgb_patch = rgb_emb[i]

                # 安全放置特征
                large_rgb_embedding[start_y:end_y, start_x:end_x, :] += rgb_patch.to(device)
                count_image[start_y:end_y, start_x:end_x, 0] += 1

            patch_count += len(coords)
            batch_count += 1

            # 清理内存
            del rgb_emb,  rgb, res, all_embedding
            torch.cuda.empty_cache()

        del dataset, dataloader
        torch.cuda.empty_cache()

    # 平均化重叠区域
    mask = count_image > 0
    large_expr_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)
    large_rgb_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)

    # 汇总处理信息
    end_time = time.time()
    total_time = end_time - start_time

    return {
            "rgb_embedding": large_rgb_embedding,
            "dimensions": (max_x, max_y)
    }

@torch.no_grad()
def process_and_infer_optimized(processer, model, patch_size=224, stride=16,
                                batch_size=32,
                                target_size=(14, 14), device='cuda'):
    """优化版本：直接将推理结果放入大图对应位置"""
    start_time = time.time()
    patch_count = 0
    batch_count = 0

    # 计算网格边界
    grid_bounds = (
        int(processer.tissue_grid['tl_xn'].min()),
        int(processer.tissue_grid['tl_xn'].max()),
        int(processer.tissue_grid['tl_yn'].min()),
        int(processer.tissue_grid['tl_yn'].max())
    )

    # 计算大图尺寸
    feature_size = target_size[0]  # 假设target_size是正方形
    max_x = grid_bounds[1]
    max_y = grid_bounds[3]
    # target_height = int(max_y / stride + feature_size)
    # target_width = int(max_x / stride + feature_size)

    # 使用更安全的尺寸计算方式
    target_height = int(np.ceil(max_y / stride)) + feature_size
    target_width = int(np.ceil(max_x / stride)) + feature_size

    # 初始化大图和计数器
    large_expr_embedding = torch.zeros((target_height, target_width, 1024),
                                       dtype=torch.float32, device=device)
    large_rgb_embedding = torch.zeros((target_height, target_width, 1024),
                                      dtype=torch.float32, device=device)
    count_image = torch.zeros((target_height, target_width, 1),
                              dtype=torch.float32, device=device)

    # 分批处理patches
    for processed_patches, patch_coordinates in collect_patches(
            processer, patch_size, stride, grid_bounds, batch_size=256
    ):
        if not processed_patches:
            continue

        dataset = IntegratedDataset(
            processed_patches=processed_patches,
            patch_coordinates=patch_coordinates,
            target_size=target_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        for batch_data in dataloader:
            data_batch,res = batch_data
            coords = data_batch["coords"]

            # 准备数据并推理
            res = res.unsqueeze(-1).unsqueeze(-1).to(device)
            rgb = data_batch["rgb"].to(device, non_blocking=True).permute(0, 3, 1, 2)
            expr = data_batch["expr"].to(device).permute(0, 3, 1, 2)

            all_embedding = model.forward_all(rgb, expr, res)

            # 分割embeddings
            mid = all_embedding.shape[1] // 2
            rgb_emb = all_embedding[:, :mid, :].view(-1, *target_size, 1024)
            expr_emb = all_embedding[:, mid:, :].view(-1, *target_size, 1024)

            # ⚠️ 关键修复5：添加边界检查的特征放置
            for i in range(len(coords)):
                x, y = coords[i]
                start_x = int(x / stride)
                start_y = int(y / stride)
                end_x = start_x + feature_size
                end_y = start_y + feature_size

                # 边界检查
                if end_x > target_width or end_y > target_height:
                    # 调整到边界内
                    end_x = min(end_x, target_width)
                    end_y = min(end_y, target_height)
                    
                    # 如果调整后区域太小，跳过
                    if end_x <= start_x or end_y <= start_y:
                        continue
                    
                    # 调整特征块大小
                    actual_height = end_y - start_y
                    actual_width = end_x - start_x
                    expr_patch = expr_emb[i][:actual_height, :actual_width, :]
                    rgb_patch = rgb_emb[i][:actual_height, :actual_width, :]
                else:
                    expr_patch = expr_emb[i]
                    rgb_patch = rgb_emb[i]

                # 安全放置特征
                large_expr_embedding[start_y:end_y, start_x:end_x, :] += expr_patch.to(device)
                large_rgb_embedding[start_y:end_y, start_x:end_x, :] += rgb_patch.to(device)
                count_image[start_y:end_y, start_x:end_x, 0] += 1

            patch_count += len(coords)
            batch_count += 1

            # 清理内存
            del rgb_emb, expr_emb, rgb, expr, res, all_embedding
            torch.cuda.empty_cache()

        del dataset, dataloader
        torch.cuda.empty_cache()

    # 平均化重叠区域
    mask = count_image > 0
    large_expr_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)
    large_rgb_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)

    return {
            "rgb_embedding": large_rgb_embedding,
            "expr_embedding":large_rgb_embedding,
            "dimensions": (max_x, max_y)
    }



def process_single_sample(mode,input_path,output_path, model, device):
    
    """Process a single tissue sample to generate feature embeddings.

    Reads input data, processes it, and generates either RGB-only or joint
    RGB-expression embeddings based on the specified mode. Handles error
    conditions and saves results to the specified output path.

    Args:
        mode (str): Processing mode, either 'rgb2emb' for RGB-only embeddings
            or 'all2emb' for combined RGB-expression embeddings
        input_path (str): Path to the input sample directory
        output_path (str): Path to save the output results
        model (torch.nn.Module): Pre-initialized STORM model
        device (str): Device to run the model on

    Returns:
        tuple[bool, str, str]: A tuple containing:
            - Boolean indicating success or failure
            - Status string ('notfound', 'failed', or None for success)
            - Error message string if failed, otherwise None
    """

    output_embedding_path = output_path+"/embeddings.pt"
    #input_path = Path(f"/lustre1/zxzeng/bwqin/STORM_main/single_section/hmdb_others/{hmid}")
    #input_path = Path(f"/lustre1/zxzeng/bwqin/STORM/Xenium/Breast_Xenium_public/pesudo_visium/")#your files
    gene_token_path ="gene_token_homologs.csv"
    try:   
        start_time = time.time()
        
        # Initialize reader and processor
        reader = VisiumReader()
        reader.read_all(
            folder_path=str(input_path),
            gene_token=str(gene_token_path),
            method="binary",
            key="symbol"
        )

        processer = VisiumPreprocesser(reader, 224)
        ''''''
        #processer.crop_img()
        #processer.generate_adata()
        os.makedirs(output_path,exist_ok=True)
        processer.save(final_grid_path=output_path+"/grid.csv",
                final_h5ad_path=output_path+"/adata.h5ad",
                final_png_path=output_path+"/tissue.png",
                final_color_path=output_path+"/raw_color.txt")
        # Process and get embeddings
        if mode == "rgb2emb":
            embeddings = process_and_infer_optimized_rgb2emb(
                processer=processer,
                model=model,
                patch_size=224,
                stride=224,
                #stride=224,
                batch_size=128,
                target_size=(14, 14),
                device=device
            )
        elif mode == "all2emb":
            embeddings = process_and_infer_optimized(
                processer=processer,
                model=model,
                patch_size=224,
                stride=224,
                #stride=224,
                batch_size=128,
                target_size=(14, 14),
                device=device
            )
        # Save results
        torch.save(embeddings, output_embedding_path)
        process_time = time.time() - start_time
        return True, None, None

    except FileNotFoundError as e:
        return False, "notfound", str(e)
    except Exception as e:
        return False, "failed", str(e)
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

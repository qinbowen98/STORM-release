import os
import cv2
import numpy as np
import openslide
import pyvips

def downsample_svs(input_svs_path, output_tiff_path, target_mpp=0.5):
    '''
    Downsample an SVS whole slide image to specified target microns per pixel (MPP) and save as TIFF
    
    Args:
        input_svs_path (str): Path to input SVS file
        output_tiff_path (str): Path to save output TIFF file
        target_mpp (float): Target resolution in microns per pixel (default: 0.5)

     Returns:
        None
    '''
    slide = openslide.OpenSlide(input_svs_path)
    current_mpp = slide.properties.get('openslide.mpp-x', None)
    if current_mpp is None:
        print(f"Warning: Missing MPP in {input_svs_path}, assuming 1.0 μm/pixel.")
        current_mpp = 1.0
    current_mpp = float(current_mpp)
    downsample_factor = target_mpp / current_mpp

    width, height = slide.dimensions
    new_width = int(width / downsample_factor)
    new_height = int(height / downsample_factor)

    level_downsamples = [float(slide.level_downsamples[i]) for i in range(slide.level_count)]
    chosen_level = None
    for i, ds in enumerate(level_downsamples):
        if ds >= downsample_factor:
            chosen_level = i
            break
    if chosen_level is None:
        chosen_level = slide.level_count - 1

    level_w, level_h = slide.level_dimensions[chosen_level]
    region = slide.read_region((0, 0), chosen_level, (level_w, level_h)).convert("RGB")
    region_np = np.array(region, dtype=np.uint8)

    actual_downsample = level_downsamples[chosen_level]
    additional_downsample = downsample_factor / actual_downsample
    if additional_downsample != 1.0:
        region_np = cv2.resize(region_np, (new_width, new_height), interpolation=cv2.INTER_AREA)

    downsampled_vips = pyvips.Image.new_from_memory(
        region_np.tobytes(), new_width, new_height, 3, 'uchar'
    )

    resolution_per_cm = 10000 / target_mpp

    downsampled_vips.write_to_file(
        output_tiff_path,
        compression='jpeg',
        tile=True,
        pyramid=True,
        xres=resolution_per_cm / 10,
        yres=resolution_per_cm / 10,
        resunit="cm"
    )

    slide.close()
    print(f"✅ Saved TIFF: {output_tiff_path}")

def process_folder(input_dir, output_dir, target_mpp=0.5):
    '''
    Batch process SVS files in a directory to downsample and convert to TIFF format
    
    Args:
        input_dir (str): Path to directory containing input SVS files
        output_dir (str): Path to directory where downsampled TIFF files will be saved
        target_mpp (float): Target microns per pixel for downsampling (default: 0.5)
    
    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".svs"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path.replace(".svs", ".tiff"))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                downsample_svs(input_path, output_path, target_mpp=target_mpp)


import os
import h5py
import numpy as np
#import openslide

def extract_tiles(tiff_folder, h5_folder, output_folder):
    '''
    Extract image tiles from TIFF images using coordinates stored in H5 files
    
    Args:
        tiff_folder (str): Path to folder containing TIFF image files
        h5_folder (str): Path to folder containing H5 files with coordinate data
        output_folder (str): Path to folder where extracted tiles will be saved

     Returns:
        None
        
    '''
    os.makedirs(output_folder, exist_ok=True)

    for h5_file in os.listdir(h5_folder):
        if not h5_file.endswith(".h5"):
            continue
        sample_name = h5_file[:-3]
        h5_path = os.path.join(h5_folder, h5_file)

        with h5py.File(h5_path, 'r') as h5:
            coords = np.array(h5['coords'])

        tiff_path = os.path.join(tiff_folder, f"{sample_name}.tiff")
        if not os.path.exists(tiff_path):
            print(f"TIFF file for {sample_name} not found, skipping...")
            continue

        sample_output_folder = os.path.join(output_folder, sample_name)
        os.makedirs(sample_output_folder, exist_ok=True)

        slide = openslide.OpenSlide(tiff_path)

        for (x, y) in coords:
            region = slide.read_region((int(x), int(y)), 0, (224, 224)).convert("RGB")

            tif_filename = f"posX_{x}_posY_{y}_{sample_name}_HE.tif"
            pt_filename = f"posX_{x}_posY_{y}_{sample_name}_expr.pt"

            region.save(os.path.join(sample_output_folder, tif_filename))

            pt_path = os.path.join(sample_output_folder, pt_filename)
            with open(pt_path, 'wb') as pt_file:
                pass

        print(f"Processed sample: {sample_name}")

    print("Processing complete.")

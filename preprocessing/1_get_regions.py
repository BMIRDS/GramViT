import argparse
import csv
import cv2
import glob
import os
import random
import openslide
import numpy as np
from pathlib import Path
import random
import shutil
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def convert_wsi_to_images(slide_path, label, fold, output_folder, target_size, level=0):
    
    # paths
    mask_folder = os.path.join(output_folder, "masks")
    region_folder = os.path.join(output_folder, "regions", fold, label, os.path.basename(slide_path).split(".svs")[0])
    thumbnail_path = os.path.join(mask_folder, os.path.basename(slide_path).split(".svs")[0] + "-thumbnail.jpg")
    overlay_path = os.path.join(mask_folder, os.path.basename(slide_path).split(".svs")[0] + "-overlay.jpg")
    mask_path = os.path.join(mask_folder, os.path.basename(slide_path).split(".svs")[0] + "-mask.npy")
    
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    if not os.path.exists(region_folder1):
        os.makedirs(region_folde)
        
    # load slide
    print("converting: {}".format(slide_path))
    slide = openslide.open_slide(slide_path)
    level_dims = slide.level_dimensions
    width = level_dims[-1][0]
    height = level_dims[-1][1]
    #load/create mask
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        
    else:
        img = slide.read_region((0,0), 2, level_dims[2])
        img = img.convert("RGB") 
        img.save(thumbnail_path)
        
    
        np_im = np.array(img)
        g = np_im[:, :, 0]
        b = np_im[:, :, 1]
        r = np_im[:, :, 2]
        
        rb_avg = (r+b)/2
        
        mask = ((r < 232)  | (b < 232)) & (rb_avg < (g-10))
        
        fig = plt.figure()
        ax1 = fig.add_axes((0, 0, 1, 1), label='thumbnail')
        ax2 = fig.add_axes((0, 0, 1, 1), label='mask')
        ax1.imshow(np_im)
        ax1.axis('off')
        ax2.imshow(mask, alpha=0.5)
        ax2.axis('off')
        plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        np.save(mask_path, mask)

    # print("extracting regions")
    
    count = 0
    w = 64
    while w < width - 64:
        h = 64
        while h < height - 64:
            region_path = os.path.join(region_folder1, os.path.basename(slide_path).split(".svs")[0] + "-" + str(w) + "-" + str(h) + ".jpg")
            if os.path.exists(region_path1):
                count = count + 1
            else:
                percent = np.mean(mask[h:h+region_size/16,w:w+region_size/16])
            
                if percent > 0.2:

                    region = slide.read_region((w*16,h*16), 0, (region_size,region_size))
                    region = region.convert("RGB")
                    
                    np_im = np.array(region)
                    cv2_im = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)
                    blur_map = cv2.Laplacian(cv2_im, cv2.CV_64F)
                    score = np.var(blur_map)
                    var = np.var(np_im)
                    
                    if (score > 60) and (var > 70):
                        region.save(region_path)
                        count = count + 1     
            h = h + region_size/16        
        w = w + region_size/16
    print(slide_path, "final count:", count)



def process_slides_from_csv(csv_file_path, output_folder, target_size, level=0, 
                            ignore_first_column=True):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    keys = []
    out = [] 


    with open(csv_file_path, newline ='') as csvfile:
        reader = csv.DictReader(csvfile)
        keys = reader.fieldnames

        for row in reader:
            temp = {} 
            for key in keys:
                value = row[key]
                temp[key] = value
            else: 
                out.append(temp)

    slide_paths = [
        item['id_patient'] for item in out
    ]
    labels = [
        item['morphology'] for item in out
    ]
    folds = [
        item['fold'] for item in out
    ]
    assert len(slide_paths) > 0


    with ProcessPoolExecutor(max_workers=1) as executor:
        for i in range(len(slide_paths)):
            slide_path = os.path.basename(slide_paths[i]).split("-")[0] + ".svs"
                executor.submit(convert_wsi_to_images, slide_path, labels[i], folds[i], output_folder, target_size, level=level)
            


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Convert slides into images")  
    parser.add_argument("csv_file_path", type=str, help="")
    parser.add_argument("output_folder", type=str, help="")
    parser.add_argument("region_size", type=int, help="")
 
    args = parser.parse_args()
    csv_file_path = args.csv_file_path
    output_folder = args.output_folder
    region_size = args.region_size

    process_slides_from_csv(csv_file_path, output_folder, region_size)

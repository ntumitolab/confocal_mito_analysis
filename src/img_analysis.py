import os
from pathlib import Path
from typing import Union, List, Dict
from argparse import ArgumentParser
from collections import namedtuple
from itertools import product

from scipy import ndimage as nd
import pandas as pd
from skan import Skeleton, summarize
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import skeletonize
from skimage.exposure import adjust_sigmoid
import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.utils import create_folder_for_each_czi, list_all_folders, list_tif_in_dir
from src.skeleton import SkeletonAnalyzer
from src.masks import get_binary_tmrm, get_binary_nucleus, get_binary_img

from tqdm import tqdm


def single_cell_mito_analysis(input_dir_path, 
                              sel_levels, 
                              fig_ext=".png", 
                              figsize=(6.4, 4.8), 
                              dpi=500):
    result_dic = {}
    sel_fields = ['Cell ID', 
                  'Total Counts of Mitochondria', 
                  'Total Mitochondrial Area', 
                  'Average Mitochondrial Area', 
                  'Average Mitochondrial Perimeter',
                  'Average Mitocondrial Solidity',
                  'Max Mitocondrial Area/Total Mitocondrial Area',
                  'Branch Number per Mitochondria',
                  'Branch Length per Mitochondria',
                  'Average Node Degree']
    
    list_of_name = list_tif_in_dir(input_dir_path, sel_levels=sel_levels)
    print(f"Found {list_of_name}")

    summary_dir = Path(input_dir_path) / "all_mito"
    summary_dir.mkdir(exist_ok=True)

    for i, file_path in tqdm(enumerate(list_of_name), total=len(list_of_name)):
        img = cv2.imread(str(file_path.resolve()), 0)
        img_b, binary2 = get_binary_img(img)
        skeleton_analyzer = SkeletonAnalyzer(binary2, tmrm_img=None, binary_nucleus=None, default_pixel_size=0.035, cell_id=file_path.stem)

        result_dic[i] = skeleton_analyzer.get_results(sel_fields)
        skeleton_analyzer.plot_skeleton(original_img=img_b, 
                                        skeleton_path=(summary_dir / "skeleton").with_suffix(fig_ext),
                                        skeleton_mito_path=(summary_dir / "skeleton mito").with_suffix(fig_ext),
                                        figsize=figsize,
                                        dpi=dpi)

    df = pd.DataFrame(result_dic).T.set_index('Cell ID')
    
    df.to_csv(summary_dir / "summary_feature.csv")
    return df


def population_mito_analysis(input_dir_path):
    list_of_name = create_folder_for_each_czi(input_dir_path)
    result_dic = {}
    sel_fields = ['Img ID', 
                  'Total Counts of Mitochondria',
                  'Average Mitochondrial Area',
                  'Average Mitochondrial Area per Cell',
                  'Average Mitochondrial Perimeter',
                  'Average Mitochondrial Perimeter per Cell',
                  'Branch Number per Mitochondria',
                  'Branch Length per Mitochondria',
                  'Branch Number per Cell',
                  'Branch Length per Cell',
                  'Average Node Degree',
                  'Average Membrane Potential']
    
    for i, folder_path in tqdm(enumerate(list_of_name), total=len(list_of_name)):
        tmrm_path = folder_path  / f"{folder_path.stem}0000.tif"
        mask_path = folder_path  / f"{folder_path.stem}mask.png"       
        nucleus_path = folder_path   / f"{folder_path.stem}nucleusmask.png"
        tmrm_img = cv2.imread(str(tmrm_path.resolve()),0)
        binary2 = get_binary_tmrm(tmrm_img, mask_img_path=mask_path)
        nucleus_img = cv2.imread(str(nucleus_path.resolve()),0)
        binary_nucleus = get_binary_nucleus(nucleus_img, mask_img_path=nucleus_path)

        skeleton_analyzer = SkeletonAnalyzer(binary2, tmrm_img, binary_nucleus, img_id=folder_path.stem)
        result_dic[i] = skeleton_analyzer.get_results(sel_fields)
    df = pd.DataFrame(result_dic).T.set_index('Img ID')
    summary_dir = Path(input_dir_path) / "all_mito"
    summary_dir.mkdir(exist_ok=True)

    df.to_csv(summary_dir / "summary_feature.csv")
    return df


def tmrm_mito_analysis(input_dir_path):
    list_of_name = list_all_folders(input_dir_path)
    result_dic = {}
    sel_fields = ['Img ID', 
                  'Average Mitochondrial Area', 
                  'Average Mitochondrial Perimeter',
                  'Branch Number per Cell',
                  'Branch Length per Cell',
                  'Average Node Degree',
                  'Average Membrane Potential']
    for i, folder_path in tqdm(enumerate(list_of_name), total=len(list_of_name)):
        tmrm_path = folder_path  / f"{folder_path.stem}0000.tif"
        mask_path = folder_path  / f"{folder_path.stem}mask.png"       
        nucleus_path = folder_path   / f"{folder_path.stem}nucleusmask.png"
        tmrm_img = cv2.imread(str(tmrm_path.resolve()),0)
        binary2 = cv2.imread(str(mask_path.resolve()),0)
        binary_nucleus = cv2.imread(str(nucleus_path.resolve()),0)

        skeleton_analyzer = SkeletonAnalyzer(binary2, tmrm_img, binary_nucleus, default_pixel_size=1, img_id=folder_path.stem)
        result_dic[i] = skeleton_analyzer.get_results(sel_fields)
    df = pd.DataFrame(result_dic).T.set_index('Img ID')
    summary_dir = Path(input_dir_path) / "all_mito"
    summary_dir.mkdir(exist_ok=True)

    df.to_csv(summary_dir / "summary_feature.csv")
    return df


def load_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--input_dir",
                            help="Input folder path")
    arg_parser.add_argument("-m", "--method",
                            choices=["tmrm",
                                     "population",
                                     "sc"])
    arg_parser.add_argument("-e", "--exp", nargs="+", default=None)
    arg_parser.add_argument("-d", "--dish", nargs="+", default=None)
    arg_parser.add_argument("-f", "--frame", nargs="+", default=None)
    arg_parser.add_argument("-y", "--dye", nargs="+", default=["mitotracker"])
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = load_args()

    if args.method.lower() == "tmrm":
        create_folder_for_each_czi(args.input_dir)
        tmrm_mito_analysis(args.input_dir)
    elif args.method.lower() == "population":
        population_mito_analysis(args.input_dir)
    elif args.method.lower() == "sc":
        if args.exp is None and args.dish is None and args.frame is None and args.dye == []:
            sel_levels = None
        else:
            sel_levels = [[f"{exp} {dish}-{frame}" for exp, dish, frame in product(args.exp, args.dish, args.frame)], args.dye]
        single_cell_mito_analysis(args.input_dir, sel_levels)
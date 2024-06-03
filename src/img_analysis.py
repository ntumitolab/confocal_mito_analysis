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
from src.utils import create_folder_for_each_czi, split_czi_to_tiffs, list_tif_in_dir, HyperParam, get_hparams_grid, SearchDomain
from src.skeleton import SkeletonAnalyzer
from src.masks import get_binary_tmrm, get_binary_nucleus, get_binary_img

from tqdm import tqdm
import json


DEFAULT_FIELDS = ['Img ID', 
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

DEFAULT_HPARAMS_DIR_PATH = Path(__file__).parent.parent / "hparams"


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


def population_mito_analysis(input_dir_path, 
                             hparam_dir_path,
                             path_dic, 
                             sel_fields):
    result_dic = {}
    hparam_dir_path = Path(hparam_dir_path)
    tmrm_params, nuc_params = {}, {}
    if (hparam_dir_path / "tmrm.json").is_file():
        with open(hparam_dir_path / "tmrm.json", "r+") as f:
            tmrm_params = json.load(f)

    if (hparam_dir_path / "nucleus.json").is_file():
        with open(hparam_dir_path / "nucleus.json", "r+") as f:
            nuc_params = json.load(f)

    for i, (img_id, folder_path) in tqdm(enumerate(path_dic.items()), total=len(path_dic)):
        tmrm_path = folder_path  / f"{folder_path.stem}0000.tif"
        nucleus_path = folder_path  / f"{folder_path.stem}0001.tif"
        mask_path = folder_path  / f"{folder_path.stem}mask.png"       
        nucleus_mask_path = folder_path   / f"{folder_path.stem}nucleusmask.png"
        tmrm_img = cv2.imread(str(tmrm_path.resolve()),0)
        nucleus_img = cv2.imread(str(nucleus_path.resolve()),0)

        if mask_path.is_file():
            tqdm.write(f"{mask_path} exists. Skipped mask calculation")
            binary_tmrm = cv2.imread(str(mask_path.resolve()),0)
        else:
            binary_tmrm = get_binary_tmrm(tmrm_img, mask_img_path=mask_path, **tmrm_params)
        if nucleus_mask_path.is_file():
            tqdm.write(f"{nucleus_mask_path} exists. Skipped mask calculation")
            binary_nucleus = cv2.imread(str(nucleus_mask_path.resolve()),0)
        else:
            binary_nucleus = get_binary_nucleus(nucleus_img, mask_img_path=nucleus_mask_path, **nuc_params)
        skeleton_analyzer = SkeletonAnalyzer(binary_tmrm, tmrm_img, binary_nucleus, 
                                             default_pixel_size=1, img_id=img_id)
        result_dic[i] = skeleton_analyzer.get_results(sel_fields)
    df = pd.DataFrame(result_dic).T.set_index('Img ID')
    summary_dir = Path(input_dir_path) / "all_mito"
    summary_dir.mkdir(exist_ok=True)
    df.to_csv(summary_dir / "summary_feature.csv")
    return df


def _prepare_hparams(n, 
                     r=None, 
                     use_tmrm=True,
                     hp_to_test="default",
                     seed=42,
                     step_like=True):
    hp_adjust_sigmoid__added_cutoff = HyperParam("added_cutoff", [1e-6, 0.1], search_domain=SearchDomain.loguniform)
    hp_adjust_sigmoid__gain = HyperParam("gain", [2, 11], search_domain=SearchDomain.intuniform)
    hp_denoise_tv_chambolle__weight = HyperParam("weight", [1e-2, 1e2], search_domain=SearchDomain.loguniform)
    hp_median_filter__filter_size = HyperParam("filter_size", [2, 7], search_domain=SearchDomain.intuniform)
    hp_dilate__dilate_kernel_size = HyperParam("dilate_kernel_size", [2, 7], search_domain=SearchDomain.intuniform)
    
    if use_tmrm:
        hp_to_test = [hp_adjust_sigmoid__added_cutoff,
                    hp_adjust_sigmoid__gain,
                    hp_denoise_tv_chambolle__weight] if hp_to_test == "default" else hp_to_test
    else:
        hp_to_test = [hp_adjust_sigmoid__added_cutoff,
                    hp_adjust_sigmoid__gain,
                    hp_median_filter__filter_size] if hp_to_test == "default" else hp_to_test
    
    return get_hparams_grid(n=n, random_k=r, seed=seed, step_like=step_like, *hp_to_test)
    

def search_mask(path_dic, 
                n, 
                r=None, 
                use_tmrm=True,
                hp_to_test="default",
                seed=42,
                step_like=True):
    hparam_grid = _prepare_hparams(n=n,
                                   r=r,
                                   use_tmrm=use_tmrm,
                                   hp_to_test=hp_to_test,
                                   seed=seed,
                                   step_like=step_like)
    
    pbar = tqdm(total=len(hparam_grid) * len(path_dic))
    for i, (img_id, folder_path) in enumerate(path_dic.items()):
        img_path = folder_path  / (f"{folder_path.stem}0000.tif" if use_tmrm else f"{folder_path.stem}0001.tif")
        mask_folder_path: Path = folder_path / ("tmrm_masks" if use_tmrm else "nucleus_masks")
        mask_folder_path.mkdir(exist_ok=True, parents=True)
        
        raw_img = cv2.imread(str(img_path.resolve()),0)

        df = pd.DataFrame(list(hparam_grid))
        df["is_best"] = "X"  # let users decide
        df.index = df.index.to_series().apply(lambda x: f"trial_{x}")
        df.to_csv(mask_folder_path / "hparam_tested.csv")

        for i, hp in enumerate(hparam_grid):
            fn = [f"{k}_{v:.2E}" for k, v in hp.items()]
            fn = "__".join(fn)
            if use_tmrm:
                get_binary_tmrm(tmrm=raw_img,
                                mask_img_path=mask_folder_path / f"trial_{i}.png",
                                **hp)
            else:
                get_binary_nucleus(nucleus=raw_img,
                                   mask_img_path=mask_folder_path / f"trial_{i}.png",
                                   **hp)
            pbar.update(1)


def sel_best_param(file_path, save_param_path, is_tmrm):
    df = pd.read_csv(file_path, index_col=0)
    best_params = df[df["is_best"] != "X"].drop(columns=["is_best"]).iloc[0, :].to_dict()
    save_param_path = Path(save_param_path)
    save_param_path.mkdir(exist_ok=True, parents=True)

    with open(save_param_path / ("tmrm.json" if is_tmrm else "nucleus.json"), "w+") as f:
        json.dump(best_params, f)


def load_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--input_dir",
                            help="Input folder/file path")
    arg_parser.add_argument("-o", "--output",
                            help="Output param file's folder")
    arg_parser.add_argument("-b", "--best_params", default=DEFAULT_HPARAMS_DIR_PATH,
                            help="Path to the folder with best parameters (.json) for performing binarization."
                                 "Default to the default parameter files")
    arg_parser.add_argument("-m", "--method",
                            choices=["population",
                                     "sc",
                                     "search_mask",
                                     "sel_best_param"],
                            help="Used method name, choices are search_mask, sel_best_param, population, and sc")
    arg_parser.add_argument("-e", "--exp", nargs="+", default=None,
                            help="")
    arg_parser.add_argument("-d", "--dish", nargs="+", default=None)
    arg_parser.add_argument("-f", "--frame", nargs="+", default=None)
    arg_parser.add_argument("-y", "--dye", nargs="+", default=["mitotracker"])
    arg_parser.add_argument("-t", "--target_czi", nargs="+", help="The czi files to be tested. Only used in search_mask method")
    arg_parser.add_argument("-n", "--n_sample", type=int, default=3, 
                            help="Size of each hparameter in the hyperparameter pool. The script will test n^(hp number) samples if -r is not specified")
    arg_parser.add_argument("-r", "--random_pick", type=int, default=10,
                            help="Number of tested hparameters. The script will test n^(hp number) samples if -r is not specified")
    arg_parser.add_argument("-u", "--use_nucleus", action="store_true",
                            help="To use nucleus as mask search target or not (if not, use TMRM images.)")
    arg_parser.add_argument("-s", "--sep_dir", action="store_true",
                            help="To save masks in separated folders for each czi file or not")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    if args.method.lower() == "population":
        path_dic=split_czi_to_tiffs(args.input_dir)
        population_mito_analysis(args.input_dir,
                                 hparam_dir_path=args.best_params,
                                 path_dic=path_dic, 
                                 sel_fields=DEFAULT_FIELDS)
    elif args.method.lower() == "sc":
        if args.exp is None and args.dish is None and args.frame is None and args.dye == []:
            sel_levels = None
        else:
            sel_levels = [[f"{exp} {dish}-{frame}" for exp, dish, frame in product(args.exp, args.dish, args.frame)], args.dye]
        single_cell_mito_analysis(args.input_dir, sel_levels)

    elif args.method.lower() == "search_mask":
        path_dic=split_czi_to_tiffs(args.input_dir, target_czi=args.target_czi,)
        search_mask(path_dic=path_dic, n=args.n_sample, r=args.random_pick, use_tmrm=not args.use_nucleus)

    elif args.method.lower() == "sel_best_param":
        sel_best_param(args.input_dir, args.output, not args.use_nucleus)
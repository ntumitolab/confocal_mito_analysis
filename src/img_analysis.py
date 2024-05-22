import os
from pathlib import Path
from typing import Union, List, Dict
from argparse import ArgumentParser

from scipy import ndimage as nd
import pandas as pd
from skan import Skeleton, summarize
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import skeletonize
from skimage.exposure import adjust_sigmoid
import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_folder_for_each_czi(folder_path: Union[str, Path],
                               created_dirs_path: Union[str, Path] = "./"):
    list_of_name = [fn.stem for fn in Path(folder_path).iterdir() if fn.suffix == ".czi"]
    for name in list_of_name:
        Path(created_dirs_path, name).mkdir(parents=True, exist_ok=True)
    return list_of_name


def tmrm_mask(folder_path: str):
    tmrm_n = '/'+folder_path.split('/')[-1] +'0000.tif'
    tmrm_path = folder_path + tmrm_n
    tmrm = cv2.imread(tmrm_path,0)
    print(tmrm_path)
    
    denoise_tmrm = denoise_tv_chambolle(tmrm, weight = 0.05)
    denoise_tmrm2 = np.uint8((denoise_tmrm*(255/np.max(denoise_tmrm))).astype(int))
    kernel = np.ones((3, 3), np.uint8)
    denoise_tmrm2 = cv2.erode(denoise_tmrm2, kernel, iterations=3)

    ret_tmrm, thresh_tmrm = cv2.threshold(denoise_tmrm2, np.min(denoise_tmrm2), np.max(denoise_tmrm2), cv2.THRESH_OTSU)
    tmrm_2 = adjust_sigmoid (denoise_tmrm2, (ret_tmrm/np.max(denoise_tmrm2))+0.04, 7)
    tmrm_2 = np.uint8(tmrm_2)
    ret_tmrm2, binary_tmrm = cv2.threshold(tmrm_2, 0, 255, cv2.THRESH_OTSU)
    
    mask_path = folder_path +'/'+folder_path.split('/')[-1] + 'mask.png'
    plt.imsave(mask_path,binary_tmrm,cmap="gray")
    return tmrm, binary_tmrm


def nucleus_mask(folder_path: str):
    nucleus_n = '/'+folder_path.split('/')[-1] +'0001.tif'
    nucleus_path = folder_path + nucleus_n
    nucleus = cv2.imread(nucleus_path,0)
    print(nucleus_path)

    denoise_nucleus = nd.median_filter(nucleus, 3)
    kernel = np.ones((3, 3), np.uint8)
    denoise_nucleus2 = cv2.dilate(denoise_nucleus, kernel, iterations=2)

    ret_nucleus, thresh_nucleus = cv2.threshold(denoise_nucleus2, np.min(denoise_nucleus2), np.max(denoise_nucleus2), cv2.THRESH_OTSU)
    nucleus_2 = adjust_sigmoid (denoise_nucleus2, (ret_tmrm/np.max(denoise_nucleus2))+0.04, 7)
    nucleus_2 = np.uint8(nucleus_2)
    ret_nucleus2, binary_nucleus = cv2.threshold(nucleus_2, 0, 255, cv2.THRESH_OTSU)
    binary_nucleus = cv2.dilate(binary_nucleus, kernel, iterations=10)
    binary_nucleus = cv2.erode(binary_nucleus, kernel, iterations=2)
    mask_path = folder_path +'/'+folder_path.split('/')[-1] + 'nucleusmask.png'
    plt.imsave(mask_path,binary_nucleus,cmap="gray")
    return binary_nucleus


def folder_all(rootpath: str) -> List[Path]:
    well_list = ['A','B','C','D','E','F','G','H']
    folder_list = [Path(rootpath, f"{w}{j}") for w in well_list for j in range(1, 13)]
    return folder_list


def particle_skeleton_analysis(binary2: np.ndarray, 
                               tmrm_img: np.ndarray, 
                               binary_nucleus: np.ndarray):
    
    pixel_size = 1
    ###### Particle Analysis ######
        
    ###----------- total mito counts -----------###
    cnts, hier = cv2.findContours(binary2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_mito_counts = len(cnts)
    
    ###----------- total nucleus counts -----------###
    cnts_n, hier_ = cv2.findContours(binary_nucleus.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    len_n = [len(i) for i in cnts_n if len(i)>10]
    total_nucleus_counts = len(len_n)

    ###----------- total mito area -----------###
    total_mito_area = len(list(np.where(binary2 == 255)[0]))*pixel_size*pixel_size
    
    ###----------- average mito area -----------###
    aver_mito_area = total_mito_area/total_nucleus_counts
    
    ###----------- average mito perimeter -----------###
    cnts_pier, hier_pier = cv2.findContours(binary2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter_list = [(cv2.arcLength(c, True)*pixel_size) for c in cnts_pier]
    aver_mito_perimeter = sum(perimeter_list)/total_nucleus_counts
    
    
    ###### Skeleton Analysis ######
    binary3 = binary2.copy()
    binary3[np.where(binary3==255)]=1
    
    skeleton = skeletonize(binary3)
    #pixel_graph, coordinates = skan.skeleton_to_csgraph(skeleton)
    branch_data = summarize(Skeleton(skeleton))
    
    ###----------- Branches num./len. per cell -----------###
    branch_len = list(branch_data['branch-distance'])
    branches_num_per_cell = len(branch_len)/total_nucleus_counts
    branches_len_per_cell = sum(branch_len)*pixel_size/total_nucleus_counts
    
    ###----------- Average node degrees -----------### 
    
    node_id_src = list(branch_data['node-id-src'])
    node_id_dst = list(branch_data['node-id-dst'])
    node_id_cnts = node_id_src + node_id_dst
    
    count_each = np.array([node_id_cnts.count(i) for i in node_id_cnts])
    degree_unique = list(count_each[list(np.unique(node_id_cnts, return_index=True)[1])])
    
    aver_node_degree = sum(degree_unique)/len(degree_unique)

    ###----------- Average membrane potential -----------### 
    average_membrane_potential = np.sum(tmrm_img[np.where(binary2==255)])/total_mito_area
    
    return(aver_mito_area,aver_mito_perimeter,branches_num_per_cell,branches_len_per_cell,aver_node_degree,average_membrane_potential)


def batch_analysis(root_path: str):
    
    list_of_name = folder_all(root_path)
    print(list_of_name[0].name)
    df = pd.DataFrame(columns=['Img_ID', 
                               'Average Mitochondrial Area', 
                               'Average Mitochondrial Perimeter',
                               'Branch Number per Cell',
                               'Branch Length per Cell',
                               'Average Node Degree',
                               'Average Membrane Potential'])
        
    for i, folder_path in enumerate(list_of_name):
        tmrm_path = folder_path + '/'+folder_path.split('/')[-1] +'0000.tif'
        mask_path = folder_path +'/'+folder_path.split('/')[-1] + 'mask.png'        
        nucleus_path = folder_path + '/'+folder_path.split('/')[-1] +'nucleusmask.png'  

        print(folder_path)
        tmrm_img = cv2.imread(tmrm_path,0)
        binary2 = cv2.imread(mask_path,0)
        binary_nucleus = cv2.imread(nucleus_path,0)
        #tmrm_img, binary2 = tmrm_mask(folder_path)
        #binary_nucleus = nucleus_mask(folder_path)
        df.loc[i] = [list_of_name[i].split('/')[-1]] + list(particle_skeleton_analysis(binary2, tmrm_img, binary_nucleus))
    
    summary_dir = root_path + '/all_mito/'
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)
        
    df = df.set_index('Img_ID') 
    df.to_csv((summary_dir+'summary_feature.csv'))
    return df


def load_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--input_dir",
    help="Input folder path")

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    create_folder_for_each_czi(args.input_dir)
    batch_analysis(args.input_dir)
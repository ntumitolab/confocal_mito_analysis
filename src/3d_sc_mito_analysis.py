import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import morphology
from skimage.morphology import skeletonize
import os
import csv
from itertools import zip_longest
import scipy
from scipy import ndimage
import pandas as pd
from skimage import morphology
import scipy.ndimage as ndi
from skimage import img_as_ubyte
from skimage.exposure import adjust_sigmoid
import skan
#from skan import skeleton_to_csgraph
from skan import Skeleton, summarize
from skan import draw
import math
from scipy import stats
import seaborn as sns
from skimage.filters import unsharp_mask
import skimage
from skimage.morphology import square
import pandas as pd


def threshold_plus(img_path):
    
    img = cv2.imread(img_path,0)
    if np.max(img) > 5:
        background = np.argmax(np.bincount(list(img[img>0].flatten())))
        result_1 = unsharp_mask(img, radius=5, amount=5)
        result_1 = result_1*255/(np.max(result_1))
        result_1 = np.uint8(result_1)
        ret, thresh = cv2.threshold(result_1[result_1>0], np.min(result_1), np.max(result_1), cv2.THRESH_OTSU)
        result_2 = result_1.copy()

        result_2[np.where(result_1<=ret)]=0
        result_1[np.where(result_1>ret)]=np.min(result_1)

        ret2, thresh2 = cv2.threshold(result_1, np.min(result_1), np.max(result_1), cv2.THRESH_OTSU)
        img2 = adjust_sigmoid (result_1, (ret2/np.max(result_1))+0.04, 7)
        img2 = np.uint8(img2)
        ret2, binary2 = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)
        #ret2, binary2 = cv2.threshold(result_1, np.min(result_1), np.max(result_1), cv2.THRESH_OTSU)
        
        binary2[np.where(result_2!=0)]=255
        binary2 = skimage.morphology.erosion(binary2)

        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(binary2)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1

        min_size = 15  
        im_result = np.zeros((binary2.shape))
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                im_result[im_with_separated_blobs == blob + 1] = 255
    else:
        im_result = np.zeros(np.shape(img))

    return  im_result, img

def img_and_mask_filenames(path):
    dir_name_list = []
    for file in os.listdir(path): 
        file_path = os.path.join(path, file) 
        if os.path.splitext(file_path)[1]!='.czi': 
            dir_name_list.append(file_path+'/')
            
    cell_ID_list = []
    cell_path_list = []
    for folder_path in dir_name_list:
        for file in os.listdir(folder_path): 
            cell_path = os.path.join(folder_path, file) 
            cell_path_list.append(cell_path+'/')
            cell_ID = cell_path.split('/')[-2] +  cell_path.split('/')[-1]
            cell_ID_list.append(cell_ID)
            #print(cell_path)
    
    mask_ID_list = []
    img_path_list = []
    mask_save_list = []   
    mask_savepath_list = [] 
    sigmoid_img_list = []
    mask_ostu_list = []
    for subfolder_path in cell_path_list:
        mask_dir = subfolder_path+'mask/'
        mask_ID_list.append(mask_dir)
        if (os.path.exists(mask_dir)!=True):
            os.makedirs(mask_dir)
        for file in os.listdir(subfolder_path): 
            img_path = os.path.join(subfolder_path, file) 
            #print((img_path.split('/')[-1])[:-4])
            if os.path.splitext(img_path)[1]=='.tif': 
                name_only = (img_path.split('/')[-1])[:-4]

                img_path_list.append(img_path)
                mask_save_list.append(img_path[:-4]+'mask.png')
                mask_savepath_list.append(mask_dir+name_only+'mask.png')
                sigmoid_img_list.append(img_path[:-4]+'mito.png')
                mask_ostu_list.append(img_path[:-4]+'mask_ostu.png')
                    
    return cell_ID_list, mask_ID_list, img_path_list, mask_save_list, mask_savepath_list, sigmoid_img_list, mask_ostu_list


def create_mask(path, final_result_path, z_axis_ratio):
    if (os.path.exists(final_result_path)!=True):
        os.makedirs(final_result_path)
            
    cell_ID_list, mask_ID_list, img_path_list, mask_save_list,mask_savepath_list, sigmoid_img_list, mask_ostu_list = img_and_mask_filenames(path)
    #print(img_path_list)
    for i in range(len(img_path_list)):
        mask, img = threshold_plus(img_path_list[i])
        #ret3, binary3 = cv2.threshold(img, 0, np.max(img), cv2.THRESH_OTSU)
        plt.imsave(mask_save_list[i],mask,cmap='gray',vmax=255,vmin=0)
        plt.imsave(mask_savepath_list[i],mask,cmap='gray',vmax=255,vmin=0)
        plt.imsave(sigmoid_img_list[i],img, cmap='gray')
        #plt.imsave(mask_ostu_list[i], binary3, cmap='gray')
    final_cell_ID = []
    for i in range(len(mask_ID_list)):
        #print(cell_ID_list[i])
        final_cell_ID.append(cell_ID_list[i])
        mask_ID_each = mask_ID_list[i]
        mask_3d_resize = []
        mask_test = []
        for file in os.listdir(mask_ID_each): 
            mask_f_path = os.path.join(mask_ID_each, file) 
            if os.path.splitext(mask_f_path)[1]=='.png': 
                mask = cv2.imread(mask_f_path,0)
                mask_3d_resize.extend([mask]*z_axis_ratio)
                mask_test.extend([np.sum(mask)]*z_axis_ratio)
        
        if (mask_test.count(0) < (len(mask_test)/2)):
            cell_3d_dir = final_result_path + cell_ID_list[i] +'/'
            if (os.path.exists(cell_3d_dir)!=True):
                os.makedirs(cell_3d_dir)
            for j in range(len(mask_3d_resize)):
                plt.imsave((cell_3d_dir+str(j)+'.png'),mask_3d_resize[j],cmap='gray',vmax=255,vmin=0)
        
    #return mask_3d_ndarray

def particle_csv(path):
    list_to_read =[]
    list_of_name =[]
    for file in os.listdir(path): 
        file_path = os.path.join(path, file) 
        if os.path.splitext(file_path)[1]=='.csv': 
            list_to_read.append(file_path)
            list_of_name.append((file_path.split('/')[-1])[:-4])
    return list_to_read,list_of_name

def skeleton_csv(path):
    list_to_read =[]
    list_of_name =[]
    for file in os.listdir(path): 
        file_path = os.path.join(path, file) 
        if os.path.splitext(file_path)[1]=='.csv': 
            name = (file_path.split('/')[-1])[:-4]
            if name.split(' ')[-1] != "Results":
                list_to_read.append(file_path)
                list_of_name.append(name.split(' ')[0])

    return list_to_read,list_of_name

def features_extraction(path_p,path_s,save_path,voxel_size):
    list_to_read,list_of_name = particle_csv(path_p)
    list_to_read_s,list_of_name_s = skeleton_csv(path_s)
    
    if (str(list_of_name)!=str(list_of_name_s)):
        print("error")
        
    total_mito_count = []
    total_mito_volume = []
    total_mito_surface = []
    max_volume_to_total_volume = []    
    average_mito_volume = []
    average_mito_surface = []
    average_sphericity = []

    branch_num_per_mito = []
    branch_len_per_mito = []
    average_node_degree = []

    for i in range(len(list_to_read)):
        df =  pd.read_csv(list_to_read[i])
        df_s = pd.read_csv(list_to_read_s[i])
        
        total_mito_count.append(len(df['Label']))
        total_mito_volume.append(sum(list(df['Volume']))*((voxel_size)**3))
        total_mito_surface.append(sum(list(df['SurfaceArea']))*((voxel_size)**2))
        max_volume_to_total_volume.append(max(list(df['Volume']))*((voxel_size)**3)/total_mito_volume[-1])
        average_mito_volume.append(total_mito_volume[-1]/total_mito_count[-1])
        average_mito_surface.append(total_mito_surface[-1]/total_mito_count[-1])
        average_sphericity.append(sum(list(df['Sphericity']))/total_mito_count[-1])

        branch_num_per_mito.append(len(list(df_s['Skeleton ID']))/total_mito_count[-1])
        branch_len_per_mito.append(sum(list(df_s['Branch length']))*voxel_size/total_mito_count[-1])
        
        x_cor = list(df_s['V1 x'])
        y_cor = list(df_s['V1 y'])

        x_cor2 = list(df_s['V2 x'])
        y_cor2 = list(df_s['V2 y'])

        node_id_cnts = []
        for i in range(len(x_cor)):
            strxy = str(x_cor[i])+ " "+str(y_cor[i])
            strxy2 = str(x_cor2[i])+ " "+str(y_cor2[i])
            node_id_cnts.append(strxy)
            node_id_cnts.append(strxy2)

        count_each = np.array([node_id_cnts.count(i) for i in node_id_cnts])
        degree_unique = list(count_each[list(np.unique(node_id_cnts, return_index=True)[1])])
        aver_node_degree = sum(degree_unique)/len(degree_unique)
        average_node_degree.append(aver_node_degree)

    df2 = pd.DataFrame({'Cell ID':list_of_name, 'Total Counts of Mitochondria':total_mito_count, 'Total Mitochondrial Volume':total_mito_volume, 'Max Mitocondrial Volume/Total Mitocondrial Volume':max_volume_to_total_volume,'Average Mitochondrial Volume':average_mito_volume, 
                        'Average Mitochondrial Surface Area':average_mito_surface ,'Average Mitocondrial Sphericity':average_sphericity,
                        'Branch Number per Mitochondria':branch_num_per_mito,'Branch Length per Mitochondria':branch_len_per_mito,'Average Node Degree':average_node_degree})
    
    df2.to_csv(save_path)
    return df2
 
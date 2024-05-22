import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import csv
from itertools import zip_longest
from scipy import ndimage
import pandas as pd
from skimage import morphology
import scipy.ndimage as ndi
from skimage import img_as_ubyte
from skimage.exposure import adjust_sigmoid
from skimage.morphology import skeletonize

from skan import Skeleton, summarize
from skan import draw
import math
from scipy import stats
import seaborn as sns


def list_dir_tif(path, list_to_read, list_to_save,list_of_name): 
    for file in os.listdir(path): 
        file_path = os.path.join(path, file) 
        if os.path.splitext(file_path)[1]=='.tif': 
            list_to_read.append(file_path)
            list_to_save.append((os.path.splitext(file_path)[0] +' mask.png'))
            list_of_name.append((file_path.split('/')[-1])[:-4])
    return list_to_read,list_to_save,list_of_name

def particle_skeleton_analysis(img2, binary2, skeleton_path, skeleton_mito_path):
    
    pixel_size = 0.035 ### micro meters
    
    ###### Particle Analysis ######
    ###----------- total mito counts -----------###
    cnts, hier = cv2.findContours(binary2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_mito_counts = len(cnts)
    
    ###----------- total mito area -----------###
    total_mito_area = len(list(np.where(binary2 == 255)[0]))*pixel_size*pixel_size
    
    ###----------- average mito area -----------###
    aver_mito_area = total_mito_area/total_mito_counts
    
    ###----------- average mito perimeter -----------###
    cnts_pier, hier_pier = cv2.findContours(binary2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter_list = [(cv2.arcLength(c, True)*pixel_size) for c in cnts_pier]
    aver_mito_perimeter = sum(perimeter_list)/len(perimeter_list)
    
    ###----------- average mito solidity -----------###
    
    imgg = np.zeros(np.shape(binary2)) 
    for c in cnts:
        cv2.fillPoly(imgg, pts =[c], color=(255,255,255))
    
    solidity_list = []
    area_list = []
    for c in cnts:
        imgg = np.zeros(np.shape(binary2)) 
        cv2.fillPoly(imgg, pts =[c], color=(255,255,255))
        imgg = np.uint8(imgg)
        mito_area = len(list(np.where((binary2 == 255)&(imgg==255))[0]))
        
        hull = cv2.convexHull(c)
        imgg_convex = np.zeros(np.shape(binary2)) 
        cv2.fillPoly(imgg_convex, pts =[hull], color=(255,255,255))
        imgg_convex = np.uint8(imgg_convex)
        
        convexhull_area = len(list(np.where(imgg_convex == 255)[0])) #M['m00']
        if (convexhull_area!=0):
            solidity_list.append((mito_area / convexhull_area))

        area_list.append(mito_area*pixel_size*pixel_size)
    aver_mito_solidity = sum(solidity_list)/len(solidity_list)
    
    ###----------- max mito area/total mito area -----------###
    max_total = max(area_list)/total_mito_area

    ###### Skeleton Analysis ######
    binary3 = binary2.copy()
    binary3[np.where(binary3==255)]=1
    
    skeleton = skeletonize(binary3)
    branch_data = summarize(Skeleton(skeleton))
    
    ###----------- Branches num./len. per mito -----------###
    branch_len = list(branch_data['branch-distance'])
    branches_num_per_mito = len(branch_len)/total_mito_counts
    branches_len_per_mito = sum(branch_len)*pixel_size/total_mito_counts
    
    ###----------- Average node degrees -----------### 
    
    node_id_src = list(branch_data['node-id-src'])
    node_id_dst = list(branch_data['node-id-dst'])
    node_id_cnts = node_id_src + node_id_dst
    
    count_each = np.array([node_id_cnts.count(i) for i in node_id_cnts])
    degree_unique = list(count_each[list(np.unique(node_id_cnts, return_index=True)[1])])
    
    aver_node_degree = sum(degree_unique)/len(degree_unique)
    
    ###### Visualize Skeleton ######
    plt.imsave(skeleton_path, skeleton,cmap='gray')
    
    fig, ax = plt.subplots()
    draw.overlay_skeleton_2d(img2, skeleton, dilate=1, axes=ax)
    plt.savefig(skeleton_mito_path,dpi=500,transparent=True)
    
    return(total_mito_counts,total_mito_area,aver_mito_area,aver_mito_perimeter,aver_mito_solidity,max_total,branches_num_per_mito,branches_len_per_mito,aver_node_degree)
    
def binary_mitochondria(root_path,inten_list, n,nn):
    dir_all = []
    df = pd.DataFrame(columns=['Cell ID', 'Total Counts of Mitochondria', 'Total Mitochondrial Area', 'Average Mitochondrial Area', 'Average Mitochondrial Perimeter','Average Mitocondrial Solidity',
                           'Max Mitocondrial Area/Total Mitocondrial Area','Branch Number per Mitochondria','Branch Length per Mitochondria','Average Node Degree'])
    for k in inten_list:
        for i in n:
            for j in nn:
                dir_all.append((root_path + k+' '+i+'-'+j+'/mitotracker/'))
                
    list_to_read = []
    list_to_save = []
    list_of_name = []
    for i in dir_all:
        list_to_read,list_to_save,list_of_name = list_dir_tif(i, list_to_read, list_to_save,list_of_name)
        
    
    summary_dir = root_path + 'all_mito/'
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)
            
    for i in range(len(list_to_read)):
        img = cv2.imread(list_to_read[i],0)
        background = np.argmax(np.bincount(list(img[img>0].flatten())))         
        ret, thresh = cv2.threshold(img[img>background], 0, 255, cv2.THRESH_OTSU)
        img2 = adjust_sigmoid (img, (ret/255)+0.04, 7)
        img2 = np.uint8(img2)
        ret2, binary2 = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)
        
        skeleton_path = summary_dir+list_of_name[i]+' skeleton.png'
        skeleton_mito_path = summary_dir+list_of_name[i]+' skeleton mito.png'
        df.loc[i] = [list_of_name[i]] + list(particle_skeleton_analysis(img2, binary2, skeleton_path, skeleton_mito_path))
        
        plt.imsave(list_to_save[i],binary2,cmap="gray",vmax=255,vmin=0)
        plt.imsave((summary_dir+list_of_name[i]+' mask.png'),binary2,cmap="gray",vmax=255,vmin=0)
        
    df = df.set_index('Cell ID') 
    df.to_csv((summary_dir+'summary_features.csv'))
    return df


import cv2
import numpy as np
import pandas as pd
from skan import Skeleton, summarize
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
from skan import draw
from enum import Enum
from typing import List


class ResultFields(Enum):
    img_id = "Img ID"
    cell_id = "Cell ID"
    total_mito_counts = "Total Counts of Mitochondria"
    aver_mito_area = 'Average Mitochondrial Area'
    aver_mito_area_per_cell = 'Average Mitochondrial Area per Cell'
    aver_mito_perimeter = 'Average Mitochondrial Perimeter'
    aver_mito_perimeter_per_cell = 'Average Mitochondrial Perimeter per Cell'
    branches_num_per_mito = 'Branch Number per Mitochondria'
    branches_len_per_mito = 'Branch Length per Mitochondria'
    branches_num_per_cell = 'Branch Number per Cell'
    branches_len_per_cell = 'Branch Length per Cell'
    aver_node_degree = 'Average Node Degree',
    average_membrane_potential = 'Average Membrane Potential'
    max_to_total_mito_area_ratio = "Max Mitocondrial Area/Total Mitocondrial Area"
    aver_mito_solidity = 'Average Mitocondrial Solidity'
    

class SkeletonAnalyzer:
    def __init__(self,
                 binary2, 
                 tmrm_img, 
                 binary_nucleus,
                 default_pixel_size=1,
                 use_cached=True,
                 **instance_kws) -> None:
        self._use_cached = use_cached
        self.binary2 = binary2
        self.tmrm_img = tmrm_img
        self.binary_nucleus = binary_nucleus
        self.default_pixel_size = default_pixel_size
        self._aver_mito_perimeter = None
        self._perimeter_list = None
        self._branch_data = None
        self._total_mito_counts = None
        self._mito_pos = None
        self._mito_area_list = None
        self._aver_mito_solidity = None
        self._instance_kws = instance_kws
        self._skeleton = None

    def get_results(self,
                    fields: List[str]) -> pd.Series:
        res_dic = {}
        for field in fields:
            if field in ResultFields:
                attr_name = field
                displayed_name = getattr(ResultFields, field).name
            else:
                try:
                    attr_name = ResultFields(field).name
                except ValueError:
                    available_names = [rf.name for rf in ResultFields]
                    available_values = [rf.value for rf in ResultFields]
                    raise ValueError(f"The field {field} is not exist. "
                                     f"Please select from {available_names}",
                                     f"Or from {available_values}")
                displayed_name = field
            res_dic[displayed_name] = getattr(self, attr_name) if attr_name not in self._instance_kws else self._instance_kws[attr_name]
        return pd.Series(res_dic)

    @property
    def total_mito_counts(self):
        if self._total_mito_counts is None:
            self._total_mito_counts, self._mito_pos = self.calc_total_mito_counts(self.binary2)
        return self._total_mito_counts
    
    @property
    def mito_area_list(self):
        if self._mito_area_list is None:
            self._aver_mito_solidity = self.calc_avg_solidity(self.binary2, self.default_pixel_size)
        return self._mito_area_list
    
    @property
    def mito_pos(self):
        if self._mito_pos is None:
            self._total_mito_counts, self._mito_pos = self.calc_total_mito_counts(self.binary2)
        return self._mito_pos
    
    @property
    def aver_mito_area(self):
        return self.calc_aver_mito_area()
    
    @property
    def aver_mito_area_per_cell(self):
        return self.calc_aver_mito_area_per_cell()

    @property
    def total_mito_area(self):
        return self.calc_total_mito_area(self.binary2, self.default_pixel_size)
    
    @property
    def total_nuc_counts(self):
        return self.calc_total_nuc_counts(self.binary_nucleus)
    
    @property
    def aver_mito_perimeter(self):
        if self._aver_mito_perimeter is None:
            self._aver_mito_perimeter, self._perimeter_list = self.calc_aver_mito_perimeter(self.binary2,
                                                                                            pixel_size=self.default_pixel_size)
        return self._aver_mito_perimeter
    
    @property
    def perimeter_list(self):
        if self._perimeter_list is None:
            self._aver_mito_perimeter, self._perimeter_list = self.calc_aver_mito_perimeter(self.binary2,
                                                                                            pixel_size=self.default_pixel_size)
        return self._perimeter_list
    
    @property
    def branch_len(self):
        if self._branch_data is None:
            self._branch_data = self.get_branch_data()
        return list(self._branch_data['branch-distance'])
    
    @property
    def branches_num_per_mito(self):
        return len(self.branch_len)/self.total_mito_counts
    
    @property
    def branches_len_per_mito(self):
        return sum(self.branch_len)*self.default_pixel_size/self.total_mito_counts
    
    @property
    def branches_num_per_cell(self):
        return len(self.branch_len)/self.total_nucleus_counts
    
    @property
    def branches_len_per_cell(self):
        return sum(self.branch_len)*self.default_pixel_size/self.total_nucleus_counts

    @staticmethod
    def calc_total_mito_counts(binary2):
        cnts, hier = cv2.findContours(binary2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        total_mito_counts = len(cnts)
        return total_mito_counts, cnts
    
    @staticmethod
    def calc_total_nuc_counts(binary_nucleus):
        cnts_n, hier_ = cv2.findContours(binary_nucleus.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        len_n = [len(i) for i in cnts_n if len(i)>10]
        total_nucleus_counts = len(len_n)
        return total_nucleus_counts
    
    @staticmethod
    def calc_total_mito_area(binary2, pixel_size=1):
        return  len(list(np.where(binary2 == 255)[0]))*pixel_size*pixel_size
    
    def calc_max_to_total_mito_area_ratio(self):
        return max(self.mito_area_list) / self.total_mito_area
    
    @staticmethod
    def calc_aver_mito_perimeter(binary2, pixel_size=1):
        cnts_pier, hier_pier = cv2.findContours(binary2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perimeter_list = [(cv2.arcLength(c, True)*pixel_size) for c in cnts_pier]
        return sum(perimeter_list)/len(perimeter_list), perimeter_list
    
    @property
    def average_membrane_potential(self):
        return self.calc_average_membrane_potential(self.tmrm_img, self.binary2)
    
    @property
    def aver_mito_perimeter_per_cell(self):
        return self.calc_aver_mito_perimeter_per_cell()
    
    @property
    def aver_mito_solidity(self):
        if self._aver_mito_solidity is None:
            self._aver_mito_solidity = self.calc_avg_solidity(self.binary2, self.default_pixel_size)
        return self._aver_mito_solidity
    
    def calc_aver_mito_area(self):
        return self.total_mito_area / self.total_mito_counts

    def calc_aver_mito_area_per_cell(self):
        return self.total_mito_area / self.total_nuc_counts
    
    def calc_aver_mito_perimeter_per_cell(self):
        return sum(self.perimeter_list)/ self.total_nucleus_counts
    
    def calc_average_membrane_potential(self, tmrm_img, binary2):
        return np.sum(tmrm_img[np.where(binary2==255)])/self.total_mito_area
    
    @property
    def skeleton(self):
        if self._skeleton is None:
            self.get_skeleton(self.binary2)
        return self._skeleton
    
    @staticmethod
    def get_skeleton(binary2):
        binary3 = binary2.copy()
        binary3[np.where(binary3==255)]=1
        skeleton = skeletonize(binary3)
        return skeleton

    def get_branch_data(self):
        #pixel_graph, coordinates = skan.skeleton_to_csgraph(skeleton)
        branch_data = summarize(Skeleton(self.skeleton))
        return branch_data
    
    def calc_avg_node_degree(self):
        node_id_src = list(self.branch_data['node-id-src'])
        node_id_dst = list(self.branch_data['node-id-dst'])
        node_id_cnts = node_id_src + node_id_dst
        
        count_each = np.array([node_id_cnts.count(i) for i in node_id_cnts])
        degree_unique = list(count_each[list(np.unique(node_id_cnts, return_index=True)[1])])
        
        return sum(degree_unique)/len(degree_unique)
    
    def calc_avg_membrane_potential(self, tmrm_img, binary2):
        return np.sum(tmrm_img[np.where(binary2==255)])/self.total_mito_area

    def calc_avg_solidity(self, binary2, pixel_size=1):
        imgg = np.zeros(np.shape(binary2)) 
        for c in self.mito_pos:
            cv2.fillPoly(imgg, pts =[c], color=(255,255,255))
        
        solidity_list = []
        area_list = []
        for c in self.mito_pos:
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

        self._mito_area_list = area_list
        return sum(solidity_list)/len(solidity_list)
    
    def plot_skeleton(self, 
                      original_img, 
                      skeleton_path=None,
                      skeleton_mito_path=None,
                      figsize=(6.4, 4.8),
                      dpi=500):
        if skeleton_path is not None:
            plt.imsave(skeleton_path, 
                    self.skeleton,
                    cmap='gray')
    
        fig, ax = plt.subplots(figsize=figsize)
        draw.overlay_skeleton_2d(original_img, 
                                 self.skeleton, dilate=1, axes=ax)
        
        if skeleton_mito_path is not None:
            plt.savefig(skeleton_mito_path,
                        dpi=dpi,
                        transparent=True)
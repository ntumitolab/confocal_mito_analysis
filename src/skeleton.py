import cv2
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize
from enum import Enum


class ResultFields(Enum):
    total_mito_counts = "Total Counts of Mitochondria"
    aver_mito_area = 'Average Mitochondrial Area'
    aver_mito_area_per_cell = 'Average Mitochondrial Area per Cell'
    aver_mito_perimeter = 'Average Mitochondrial Perimeter'
    aver_mito_perimeter_per_cell = 'Average Mitochondrial Perimeter per Cell'



class SkeletonAnalyzer:
    def __init__(self,
                 binary2, 
                 tmrm_img, 
                 binary_nucleus,
                 default_pixel_size=1,
                 use_cached=True) -> None:
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

    @property
    def total_mito_counts(self):
        if self._total_mito_counts is None:
            self._total_mito_counts, self._mito_pos = self.calc_total_mito_counts(self.binary2)
        return self._total_mito_counts
    
    @property
    def mito_pos(self):
        if self._mito_pos is None:
            self._total_mito_counts, self._mito_pos = self.calc_total_mito_counts(self.binary2)
        return self._mito_pos

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
            self._branch_data = self.get_branch_data(self.binary2)
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
    
    @staticmethod
    def calc_aver_mito_perimeter(binary2, pixel_size=1):
        cnts_pier, hier_pier = cv2.findContours(binary2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perimeter_list = [(cv2.arcLength(c, True)*pixel_size) for c in cnts_pier]
        return sum(perimeter_list)/len(perimeter_list), perimeter_list
    
    def calc_aver_mito_area(self):
        return self.total_mito_area / self.total_mito_counts

    def calc_aver_mito_area_per_cell(self):
        return self.total_mito_area / self.total_nuc_counts
    
    def calc_aver_mito_perimeter_per_cell(self):
        return sum(self.perimeter_list)/ self.total_nucleus_counts
    
    @staticmethod
    def get_branch_data(binary2):
        binary3 = binary2.copy()
        binary3[np.where(binary3==255)]=1
        
        skeleton = skeletonize(binary3)
        #pixel_graph, coordinates = skan.skeleton_to_csgraph(skeleton)
        branch_data = summarize(Skeleton(skeleton))
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
        return sum(solidity_list)/len(solidity_list)
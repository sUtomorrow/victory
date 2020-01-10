# -*- coding: utf-8 -*-
# @Time     : 7/4/19 4:04 PM
# @Author   : lty
# @File     : merge_result

import math
import numpy as np
import time

def run_time(func):
    def warpper(*arg, **kw):
        local_time = time.time()
        ret = func(*arg, **kw)
        print('Function %s run time: %.2f' % (func.__name__, time.time() - local_time))
        return ret
    return warpper

################################# merge adjacent point ################################

class ClassSet(object):
    # @run_time
    def __init__(self, poi_list, max_dist=10000, threshold_z=5):
        self.class_list = []
        self.poi_list = poi_list
        idx = 0
        for _ in poi_list:
            self.class_list.append([idx])
            idx += 1

        self.poi_num = idx
        self.class_num = idx
        self.max_dist = max_dist
        self.threshold_z = threshold_z
        assert (self.class_num == self.poi_num)
        self.init_class_dist_mat()

    def init_class_dist_mat(self):
        coord_array = np.array([[point['coordX'], point['coordY'], point['coordZ']] for point in self.poi_list])
        coord_array_y = np.expand_dims(coord_array, 0)
        coord_array_x = np.expand_dims(coord_array, 1)
        coord_array_y = np.tile(coord_array_y, (self.poi_num, 1, 1))
        coord_array_x = np.tile(coord_array_x, (1, self.poi_num, 1))

        xg, yg = np.meshgrid(list(range(self.poi_num)), list(range(self.poi_num)))

        power_dis = np.power(coord_array_y - coord_array_x, 2)
        self.class_dist_mat = np.sqrt(np.sum(power_dis, axis=-1))
        self.class_dist_mat[power_dis[:, :, 2] <= self.threshold_z ** 2] = 10000 # z轴上距离小于6mm的距离设置为最大,防止z轴距离相近的进行合并
        self.class_dist_mat += (xg <= yg).astype(np.uint8) * self.max_dist

        # print(self.class_dist_mat.shape)
    # @run_time
    # def poi_dist(self, a_idx, b_idx):
    #     a = self.poi_list[a_idx]
    #     b = self.poi_list[b_idx]
    #     return math.sqrt((a['coordX'] - b['coordX']) ** 2 + (a['coordY'] - b['coordY']) ** 2 + (a['coordZ'] - b['coordZ']) ** 2)

    # @run_time
    def find_min_set_dist(self):
        min_i = -1
        min_j = -1
        min_dist = self.max_dist
        if self.class_num <= 1:
            return min_i, min_j, min_dist
        min_dist = np.min(self.class_dist_mat)  # self.max_dist
        ys, xs = np.where(self.class_dist_mat == min_dist)
        min_i = ys[0]
        min_j = xs[0]
        return min_i, min_j, min_dist

    # @run_time
    def merge_class(self, class_idx1, class_idx2):
        if class_idx1 == class_idx2:
            return
        if class_idx1 > class_idx2:
            temp = class_idx2
            class_idx2 = class_idx1
            class_idx1 = temp
        self.class_list[class_idx1] += self.class_list[class_idx2]

        assert (class_idx1 < class_idx2)

        for class_idx in range(self.class_num):
            if class_idx != class_idx1 and class_idx != class_idx2:
                if class_idx > class_idx1:
                    self.class_dist_mat[class_idx1][class_idx] = min(self.class_dist_mat[class_idx1][class_idx],
                                                                     self.class_dist_mat[class_idx2][
                                                                         class_idx] if class_idx2 < class_idx else
                                                                     self.class_dist_mat[class_idx][class_idx2])
                else:
                    self.class_dist_mat[class_idx][class_idx1] = min(self.class_dist_mat[class_idx][class_idx1],
                                                                     self.class_dist_mat[class_idx][class_idx2])

        self.class_dist_mat = np.delete(self.class_dist_mat, class_idx2, axis=1)
        self.class_dist_mat = np.delete(self.class_dist_mat, class_idx2, axis=0)
        self.class_num -= 1
        del (self.class_list[class_idx2])
        # print(len(self.class_list))
        return

    # @run_time
    def get_classes(self):
        real_class_list = []
        for c in self.class_list:
            poi_list = []
            for poi_idx in c:
                poi_list.append(self.poi_list[poi_idx])
            real_class_list.append(poi_list)
        return real_class_list

# @run_time
def agnes(poi_list, threshold_d, threshold_z, max_dist=10000):
    class_set = ClassSet(poi_list, max_dist=max_dist, threshold_z=threshold_z)
    i, j, min_d = class_set.find_min_set_dist()
    while threshold_d > min_d and class_set.class_num > 1:
        class_set.merge_class(i, j)
        i, j, min_d = class_set.find_min_set_dist()

    return class_set.get_classes()

if __name__ == '__main__':
    poi_list = []
    for i in range(10):
        poi_list.append({
            'coordX': np.random.randint(100),
            'coordY': np.random.randint(100),
            'coordZ': np.random.randint(100),
        })
    print(agnes(poi_list, threshold_d=10, max_dist=10000, threshold_z=10))

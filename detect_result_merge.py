# -*- coding: utf-8 -*-
# @Time     : 7/10/19 9:50 AM
# @Author   : ldy
# @File     : 8.merge_result
import os
import numpy as np
import pickle
import cv2
from copy import deepcopy
import pandas as pd
from AGNES import agnes

CLASS_NUM     = 4

CLASS_IDX_TO_MODEL_CLASS_ID = {
    '0': 1,
    '1': 5,
    '2': 31,
    '3': 32
}
def coord_pix_to_world(pix_coord, spacing, origin, direction):
    '''
    world coordinate: x
    direction matrix: D
    voxel coordinate: v
    spacing         : s
    origin          : o
    x = D * s * v + o
    '''
    pix_coord = np.array(pix_coord)
    direction = np.array(direction).reshape((3, 3))
    origin = np.array(origin)
    spacing = np.array(spacing)

    # the coordinate is row vector, use transform
    world_coord = np.matmul((spacing * pix_coord), direction.T) + origin
    return world_coord


def coord_world_to_pix(world_coord, direction, origin, spacing):
    '''
    world coordinate: x
    direction matrix: D
    voxel coordinate: v
    spacing         : s
    origin          : o
    x = D * s * v + o => v = D^(-1) * (x - o) / s
    '''
    world_coord = np.array(world_coord)
    direction   = np.array(direction).reshape((3, 3))
    origin      = np.array(origin)
    spacing     = np.array(spacing)

    # the coordinate is row vector, use transform
    pix_coord = np.matmul((world_coord - origin), np.linalg.inv(direction).T) / spacing
    return pix_coord


def merge_adjacent_candidate_agens(result_path, data_dir, agnes_distance_threshold=7.):
    '''合并相邻的检测点'''

    # 读取检测结果
    result_df = pd.read_csv(result_path, dtype={'seriesuid': str, "coordX": float, "coordY": float, "coordZ": float, "class": int, 'probability': float})
    # 新的检测结果
    new_result_dict = {"seriesuid": [], "coordX": [], "coordY": [], "coordZ": [], "class": [], 'probability': []}
    # 获取检测结果里面所有的seriesuid
    seriesuid_list = list(result_df['seriesuid'].unique())
    for seriesuid in seriesuid_list:
        # 对每一个seriesuid进行合并相邻点的操作
        print('merge', seriesuid)
        with open(os.path.join(data_dir, seriesuid + '.pkl'), 'rb') as fp:
            patient_info = pickle.load(fp)
        patient_result_df = result_df[result_df['seriesuid'] == seriesuid]
        for class_idx in range(CLASS_NUM):
            # 对每个类别分开进行合并
            patient_label_result_df = patient_result_df[patient_result_df['class']==CLASS_IDX_TO_MODEL_CLASS_ID[str(class_idx)]]

            # 初始化所有的检测点
            point_list = []
            for _, row in patient_label_result_df.iterrows():
                # 使用像素距离来进行合并
                row['coordX'] /= patient_info['spacing'][0]
                row['coordY'] /= patient_info['spacing'][1]
                row['coordZ'] /= patient_info['spacing'][2]
                point_list.append(row)
            if len(point_list):
                # 使用agnes算法进行相邻点的合并,合并后得到的是点的集合
                point_class_list = agnes(point_list, agnes_distance_threshold, 0)

                total_num = 0
                for point_class in point_class_list:
                    # 处理每个点集合,将一个点集合合并为一个点
                    max_prob = 0
                    coordX = coordY = coordZ = 0
                    num = 0
                    for point in point_class:
                        # 对于一个集合里面的点,进行合并操作
                        num += 1
                        # 概率取点集合里面所有点的最大值
                        if point['probability'] > max_prob:
                            max_prob = point['probability']
                            coordX = point['coordX'] * patient_info['spacing'][0]
                            coordY = point['coordY'] * patient_info['spacing'][1]
                            coordZ = point['coordZ'] * patient_info['spacing'][2]
                    total_num += num - 1
                    new_result_dict['seriesuid'].append(seriesuid)
                    new_result_dict['coordX'].append(coordX)
                    new_result_dict['coordY'].append(coordY)
                    new_result_dict['coordZ'].append(coordZ)
                    new_result_dict['class'].append(CLASS_IDX_TO_MODEL_CLASS_ID[str(class_idx)])
                    new_result_dict['probability'].append(max_prob)
    # 保存为新的结果文件
    merge_result = pd.DataFrame(new_result_dict, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
    result_save_path = result_path.replace('.csv', '_merge_pix_dis%.2f.csv' % agnes_distance_threshold)
    merge_result.to_csv(result_save_path, index=False)
    return merge_result, result_save_path

def get_effective_mask(lung_mask):
    '''两个肺的外接矩形掩码'''
    effective_mask = deepcopy(lung_mask) # zyx
    xmin = ymin = 100000
    xmax = ymax = 0
    for slice_idx in range(len(effective_mask)):
        contours, _ = cv2.findContours((effective_mask[slice_idx] > 0).astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        for contout in contours:
            x1, y1 = contout.min(axis=(0, 1))
            x2, y2 = contout.max(axis=(0, 1))
            if x1 < xmin:
                xmin = x1
            if y1 < ymin:
                ymin = y1
            if x2 > xmax:
                xmax = x2
            if y2 > ymax:
                ymax = y2
    effective_mask[:, ymin:ymax, xmin:xmax] = 1
    return effective_mask

def filter_result_by_lung_mask(result_path, data_dir, lung_mask_dir):
    '''通过肺部掩码,筛选一部分数据'''
    # 读取检测结果
    result_df = pd.read_csv(result_path, dtype={'seriesuid': str, "coordX": float, "coordY": float, "coordZ": float, "class": int, 'probability': float})
    # 新的检测结果
    new_result_dict = {"seriesuid": [], "coordX": [], "coordY": [], "coordZ": [], "class": [], 'probability': []}
    # 所有的seriesuid
    seriesuid_list = list(result_df['seriesuid'].unique())

    for seriesuid in seriesuid_list:
        print('filter', seriesuid)
        # 逐个判断每个病人的检测结果
        patient_result_df = result_df[result_df['seriesuid'] == seriesuid]
        mask = np.load(os.path.join(lung_mask_dir, seriesuid + '.npy'))
        with open(os.path.join(data_dir, seriesuid + '.pkl'), 'rb') as fp:
            patient_info = pickle.load(fp)
        # 找到病人的有效区域掩码
        mask = get_effective_mask(mask)

        for _, row in patient_result_df.iterrows():
            # 逐个进行判断
            if float(row['probability']) <= 0.2:
                continue

            # 结果中的坐标是世界坐标,需要转成像素坐标
            # print('mask shape ',mask.shape)
            pix_coord = coord_world_to_pix([row['coordX'], row['coordY'], row['coordZ']], patient_info['direction'], patient_info['origin'], patient_info['spacing'])
            # print(pix_coord)
            if int(round(pix_coord[2])) >= mask.shape[0] or int(round(pix_coord[1])) >= mask.shape[1] or  int(round(pix_coord[0])) >= mask.shape[2]:
                continue
            if mask[int(round(pix_coord[2]))][int(round(pix_coord[1]))][int(round(pix_coord[0]))] > 0:
                # 在有效区域的检测点,才添加到最终结果
                new_result_dict['seriesuid'].append(seriesuid)
                new_result_dict['coordX'].append(row['coordX'])
                new_result_dict['coordY'].append(row['coordY'])
                new_result_dict['coordZ'].append(row['coordZ'])
                new_result_dict['class'].append(row['class'])
                new_result_dict['probability'].append(row['probability'])
    # 保存结果
    filter_result = pd.DataFrame(new_result_dict, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability'])
    result_save_path = result_path.replace('.csv', '_lung_mask.csv')
    filter_result.to_csv(result_save_path, index=False)
    return filter_result, result_save_path

if __name__ == '__main__':
    import time
    # 使用掩码筛选预测点
    # TEST_DATA_DIR = '/mnt/data3/victory/rescale_z_test_dataset/'
    # TEST_DATA_DIR = '/mnt/data4/lty/victory_data/test_dataset_spacing662/'
    TEST_DATA_DIR = '/mnt/data4/lty/victory_data/testB_dataset_spacing662'
    # TEST_DATA_DIR = '/mnt/data3/victory/test_dataset/'
    # lung_mask_dir = '/mnt/data3/victory/lung_mask/'
    result_csv_path = '/mnt/data4/lty/victory/result_dir/retinanet_resnet101_C2345_small_anchor_alpha8_spacing662_const_scale_data_test0.3718/testB_resnet101_csv_18_nms_0.6_score_threshold0.2.csv'

    while True:
        if not os.path.exists(result_csv_path):
            print('wait result generate')
            time.sleep(30)
        else:
            break

    # result_csv_path = './result_dir/retinanet_resnet101_C2345_small_anchor/resnet101_csv_23_lung_mask.csv'
    # _, result_csv_path = filter_result_by_lung_mask(result_csv_path, TEST_DATA_DIR, lung_mask_dir)
    # result_csv_path = '/mnt/data4/lty/victory/result_dir/retinanet_resnet101_C2345_small_anchor/resnet101_csv_03_lung_mask.csv'
    # 合并结果文件中的相邻点
    _, merge_result_path = merge_adjacent_candidate_agens(result_csv_path, TEST_DATA_DIR, agnes_distance_threshold=7.)
# -*- coding: utf-8 -*-
# @Time     : 7/1/19 7:20 PM
# @Author   : lty
# @File     : data_preprocess

import os
from glob import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd
import pickle
import cv2
from scipy import ndimage
from copy import deepcopy


cfg = {
    'data_dir'            : '/mnt/data3/victory/tianchi_dataset/', # 数据下载目录,这个目录下面应该有testA, train_part1, train_part2, train_part3, train_part4, train_part5, 五个子文件夹以及标签文件:chestCT_round1_annotation.csv
    'target_train_dir'    : '/mnt/data4/lty/victory_data/train_dataset_spacing662/', # 训练集HU值矩阵npy以及病人信息pkl的保存文件夹
    'target_test_dir'     : '/mnt/data4/lty/victory_data/test_dataset_spacing662/', # 测试集HU值矩阵npy以及病人信息pkl的保存文件夹
    'data_split_info_dir' : '/mnt/data4/lty/victory_data/', # 数据划分信息保存文件夹
    'target_spacing_xyz'  : [0.6, 0.6, 2.0], #是否进行spacing变换,如果为None则不进行,否则应指定目标spacing
}


def resample(img_array, spacing_xyz, target_spacing_xyz):
    '''img_array: zyx'''

    spacing_zyx        = np.array(list(reversed(spacing_xyz)))
    target_spacing_zyx = np.array(list(reversed(target_spacing_xyz)))

    resize_factor  = spacing_zyx / target_spacing_zyx
    new_real_shape = img_array.shape * resize_factor
    new_shape      = np.round(new_real_shape)
    
    real_resize_factor = new_shape / img_array.shape
    target_spacing_zyx = spacing_zyx / real_resize_factor

    img_array = ndimage.interpolation.zoom(img_array, real_resize_factor, mode='constant', cval=0)
    return img_array, target_spacing_zyx[::-1] # zyx => xyz


def resample_v2(img_array, spacing_xyz, target_spacing_xyz):
    '''img_array: zyx'''

    spacing_zyx        = np.array(list(reversed(spacing_xyz)))
    target_spacing_zyx = np.array(list(reversed(target_spacing_xyz)))

    resize_factor  = spacing_zyx / target_spacing_zyx
    new_real_shape = img_array.shape * resize_factor
    new_shape      = np.round(new_real_shape)
    
    real_resize_factor = new_shape / img_array.shape
    target_spacing_zyx = spacing_zyx / real_resize_factor

    img_array = np.transpose(img_array, [1, 2, 0]) # zyx => yxz

    if img_array.shape[2] > 512:
        slice_list = [img_array[:, :, i * 512 : i * 512 + 512] for i in range((img_array.shape[2] + 511) // 512)]
        for slice_idx in range(len(slice_list)):
            slice_list[slice_idx] = cv2.resize(slice_list[slice_idx], dsize=None, fx=real_resize_factor[2], fy=real_resize_factor[1], interpolation=cv2.INTER_CUBIC)
            if len(slice_list[slice_idx].shape) == 2:
                slice_list[slice_idx] = slice_list[slice_idx][..., np.newaxis]
        img_array = np.concatenate(slice_list, axis=-1)
    else:
        img_array = cv2.resize(img_array, dsize=None, fx=real_resize_factor[2], fy=real_resize_factor[1], interpolation=cv2.INTER_CUBIC)
    
    img_array = np.transpose(img_array, [2, 0, 1]) # yxz => zyx
    if img_array.shape[2] > 512:
        slice_list = [img_array[:, :, i * 512 : i * 512 + 512] for i in range((img_array.shape[2] + 511) // 512)]
        for slice_idx in range(len(slice_list)):
            slice_list[slice_idx] = cv2.resize(slice_list[slice_idx], dsize=None, fx=1, fy=real_resize_factor[0], interpolation=cv2.INTER_CUBIC)
            if len(slice_list[slice_idx].shape) == 2:
                slice_list[slice_idx] = slice_list[slice_idx][..., np.newaxis]
        img_array = np.concatenate(slice_list, axis=-1)
    else:
        img_array = cv2.resize(img_array, dsize=None, fx=1, fy=real_resize_factor[0], interpolation=cv2.INTER_CUBIC)
    
    return img_array, target_spacing_zyx[::-1] # zyx => xyz


def resample_v3(img_array, spacing_xyz, target_spacing_xyz):
    '''img_array: zyx'''

    spacing_zyx        = np.array(list(reversed(spacing_xyz)))
    target_spacing_zyx = np.array(list(reversed(target_spacing_xyz)))

    resize_factor  = spacing_zyx / target_spacing_zyx
    new_real_shape = img_array.shape * resize_factor
    new_shape      = np.round(new_real_shape)
    
    real_resize_factor = new_shape / img_array.shape
    target_spacing_zyx = spacing_zyx / real_resize_factor

    img_array = np.transpose(img_array, [1, 2, 0]) # zyx => yxz

    slice_list = [img_array[:, :, i * 3 : i * 3 + 3] for i in range((img_array.shape[2] + 2) // 3)]
    for slice_idx in range(len(slice_list)):
        slice_list[slice_idx] = cv2.resize(slice_list[slice_idx], dsize=None, fx=real_resize_factor[2], fy=real_resize_factor[1], interpolation=cv2.INTER_CUBIC)
        if len(slice_list[slice_idx].shape) == 2:
            slice_list[slice_idx] = slice_list[slice_idx][..., np.newaxis]
    img_array = np.concatenate(slice_list, axis=-1)
    
    img_array = np.transpose(img_array, [2, 0, 1]) # yxz => zyx
    slice_list = [img_array[:, :, i * 3 : i * 3 + 3] for i in range((img_array.shape[2] + 2) // 3)]
    for slice_idx in range(len(slice_list)):
        slice_list[slice_idx] = cv2.resize(slice_list[slice_idx], dsize=None, fx=1, fy=real_resize_factor[0], interpolation=cv2.INTER_CUBIC)
        if len(slice_list[slice_idx].shape) == 2:
            slice_list[slice_idx] = slice_list[slice_idx][..., np.newaxis]
    img_array = np.concatenate(slice_list, axis=-1)
    
    return img_array, target_spacing_zyx[::-1] # zyx => xyz


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


def preprocess_patient(mhd_path, seriesuid, annotations, target_dir=None, target_spacing_xyz=None):
    mhd_file = sitk.ReadImage(mhd_path)

    img_array = sitk.GetArrayFromImage(mhd_file).astype(np.int16) # zyx

    info_pkl_dict = {}

    spacing_xyz = np.array(mhd_file.GetSpacing())
    direction   = np.array(mhd_file.GetDirection())
    origin_xyz  = np.array(mhd_file.GetOrigin())

    if target_spacing_xyz is not None:
        # 调整z轴层厚
        img_array, spacing_xyz = resample(img_array, spacing_xyz, target_spacing_xyz)
        # # print(img_array[-1, 200:300, 200:300])
        # cv2.imshow('slice_img', (np.clip(img_array[-1, :, :], -1000, 400) + 1000) / 1400)
        # cv2.waitKey()
        # exit()
        # for slice_img in img_array:
        #     cv2.imshow('slice_img', (np.clip(slice_img, -1000, 400) + 1000) / 1400)
        #     cv2.waitKey()
        # exit()
        # print('0')
        # img_array1, spacing_xyz1 = resample(deepcopy(img_array), spacing_xyz, target_spacing_xyz)
        # print('1')
        # img_array2, spacing_xyz2 = resample_v2(deepcopy(img_array), spacing_xyz, target_spacing_xyz)
        # print('2')
        # img_array3, spacing_xyz3 = resample_v3(deepcopy(img_array), spacing_xyz, target_spacing_xyz)
        # print('3')
        # for slice1, slice2, slice3 in zip(img_array1, img_array2, img_array3):
        #     cv2.imshow('slice1', (np.clip(slice1, -1000, 400) + 1000) / 1400)
        #     cv2.imshow('slice2', (np.clip(slice2, -1000, 400) + 1000) / 1400)
        #     cv2.imshow('slice3', (np.clip(slice3, -1000, 400) + 1000) / 1400)
        #     cv2.waitKey()
        # exit()

    info_pkl_dict['spacing']     = spacing_xyz
    info_pkl_dict['direction']   = direction
    info_pkl_dict['origin']      = origin_xyz
    info_pkl_dict['annotations'] = []

    if annotations is not None:
        for idx, row in annotations.iterrows():
            pix_coord = coord_world_to_pix([float(row['coordX']), float(row['coordY']), float(row['coordZ'])], direction, origin_xyz, spacing_xyz)
            x  = pix_coord[0]
            y  = pix_coord[1]
            z  = pix_coord[2]
            dx = float(row['diameterX']) / spacing_xyz[0]
            dy = float(row['diameterY']) / spacing_xyz[1]
            dz = float(row['diameterZ']) / spacing_xyz[2]
            info_pkl_dict['annotations'].append([x, y, z, dx, dy, dz, int(row['label'])])
    if target_dir is not None:
        img_save_path = os.path.join(target_dir, seriesuid + '.npy')
        pkl_save_path = os.path.join(target_dir, seriesuid + '.pkl')
        np.save(img_save_path, img_array)
        with open(pkl_save_path, 'wb') as fp:
            pickle.dump(info_pkl_dict, fp)
    return img_array, info_pkl_dict


def preprocess(data_dir, train_target_dir, test_target_dir, target_spacing_xyz=None):
    # if not os.path.exists(train_target_dir):
    #     os.mkdir(train_target_dir)

    # if not os.path.exists(test_target_dir):
    #     os.mkdir(test_target_dir)

    # 处理训练数据
    annotation_csv_path = os.path.join(data_dir, 'chestCT_round1_annotation.csv')
    annotations = pd.read_csv(annotation_csv_path, dtype=str)
    mhd_idx = 0
    for data_idx in range(1, 6):
        subset_data_dir = os.path.join(data_dir, 'train_part%d' % data_idx)
        # 所有的mhd路径
        mhd_path_list = glob(os.path.join(subset_data_dir, '*.mhd'))
        np.random.shuffle(mhd_path_list)
        for mhd_path in mhd_path_list:
            print('train mhd %d:' % mhd_idx, mhd_path)
            mhd_idx += 1
            seriesuid = os.path.basename(mhd_path).split('.mhd')[0]
            patient_annotation = annotations[annotations['seriesuid'] == seriesuid]
            preprocess_patient(mhd_path, seriesuid, patient_annotation, train_target_dir, target_spacing_xyz)

    # 处理测试集数据
    test_data_dir = os.path.join(data_dir, 'testA')
    # 所有的mhd路径
    test_mhd_path_list = glob(os.path.join(test_data_dir, '*.mhd'))
    mhd_idx = 0
    for test_mhd_path in test_mhd_path_list:
        seriesuid = os.path.basename(test_mhd_path).split('.mhd')[0]
        print('test mhd %d:' % mhd_idx, test_mhd_path)
        mhd_idx += 1
        preprocess_patient(test_mhd_path, seriesuid, None, test_target_dir, target_spacing_xyz)


def data_split(data_dir, dataset_info_save_dir, valid_ratio=0.2, seed=10086):
    np.random.seed(seed)

    seriesuid_list = sorted([x.split('.npy')[0] for x in os.listdir(data_dir) if '.npy' in x])

    np.random.shuffle(seriesuid_list)

    valid_seriesuid_num = int(len(seriesuid_list) * valid_ratio)

    valid_seriesuid_list = seriesuid_list[:valid_seriesuid_num]
    train_seriesuid_list = seriesuid_list[valid_seriesuid_num:]

    dataset_info_save_path = os.path.join(dataset_info_save_dir, 'data_split.pkl')

    with open(dataset_info_save_path, 'wb') as fp:
        pickle.dump({'train':train_seriesuid_list, 'valid': valid_seriesuid_list}, fp)


if __name__ == '__main__':
    # 数据预处理
    # preprocess(cfg['data_dir'], cfg['target_train_dir'], cfg['target_test_dir'], cfg['target_spacing_xyz'])

    # 划分训练集和验证集,只需要运行一次
    # data_split(cfg['target_train_dir'], cfg['data_split_info_dir'], valid_ratio=0.2)

    # testB数据
    test_mhd_path_list = glob(os.path.join('/mnt/data3/victory/chestCT_round1_testB/', '*.mhd'))
    mhd_idx = 0
    for test_mhd_path in test_mhd_path_list:
        seriesuid = os.path.basename(test_mhd_path).split('.mhd')[0]
        print('test mhd %d:' % mhd_idx, test_mhd_path)
        mhd_idx += 1
        preprocess_patient(test_mhd_path, seriesuid, None, '/mnt/data4/lty/victory_data/testB_dataset_spacing662', cfg['target_spacing_xyz'])
    
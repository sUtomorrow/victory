import numpy as np
import pandas as pd
import pickle
import copy
import os
import cv2
import tensorflow as tf
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image, compute_resize_scale
from keras_retinanet.utils.visualization import draw_detections


CLASS_IDX_TO_MODEL_CLASS_ID = {
    '0': 1,
    '1': 5,
    '2': 31,
    '3': 32
}

INPUT_SLICE = 3

RESULT_CSV_COLUMNS = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']

TEST_DATA_DIR = '/mnt/data4/lty/victory_data/test_dataset_spacing662/'

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


def predictTest(model, data_dir, target_path, score_threshold=0.2, seriesuids=None, imshow=True, input_min_max_side=(768, 768)):
    data_npy_name_list = [file_name for file_name in os.listdir(data_dir) if '.npy' in file_name]
    anno_list = []

    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for data_npy_name in data_npy_name_list:

        seriesuid = data_npy_name.split('.npy')[0]
        if seriesuids is not None and seriesuid not in seriesuids:
            continue
        print('detect', seriesuid)
        data_path     = os.path.join(data_dir, data_npy_name)
        info_pkl_path = os.path.join(data_dir, seriesuid + '.pkl')
        
        with open(info_pkl_path, 'rb') as fp:
            patient_info = pickle.load(fp)
        
        data_npy = np.load(data_path) # zyx
        data_npy = np.transpose(data_npy,(1,2,0)) # zyx -> yxz
        
        slice_start = INPUT_SLICE // 2

        for slice_idx in range(slice_start, data_npy.shape[2] - INPUT_SLICE // 2, max(INPUT_SLICE // 2, 1)):
            s_start   = slice_idx - INPUT_SLICE // 2
            input_data = (data_npy[:, :, s_start : s_start + INPUT_SLICE]).copy()
            input_img, scale = resize_image(preprocess_image(input_data, 'caffe'), input_min_max_side[0], input_min_max_side[1])
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(input_img, axis=0))
            boxes  = boxes[0]
            scores = scores[0]
            labels = labels[0]
            
            if imshow:
                img_show = np.tile(((input_img + 128)[:, :, 1:2]).astype(np.uint8), (1, 1, 3))
                draw_detections(img_show, boxes, scores, labels, color=(0, 255, 0), score_threshold=score_threshold)
                cv2.imshow('detect_image', img_show)
                k = cv2.waitKey()
                if k == ord('q'):
                    exit()
                elif k == ord('c'):
                    imshow = False

            boxes /= scale

            for label, score, box in zip(labels, scores, boxes):
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = box

                voxelX = (x1 + x2) / 2
                voxelY = (y1 + y2) / 2
                voxelZ = slice_idx
                
                coordX, coordY, coordZ = coord_pix_to_world([voxelX, voxelY, voxelZ], patient_info['spacing'], patient_info['origin'], patient_info['direction'])
                
                the_label = CLASS_IDX_TO_MODEL_CLASS_ID[str(label)]
                anno_list.append([seriesuid, coordX, coordY, coordZ, the_label, score])

    anno_csv = pd.DataFrame(data=anno_list, columns=RESULT_CSV_COLUMNS)
    anno_csv.to_csv(target_path, index=False, encoding='UTF-8')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    TEST_DATA_DIR = '/mnt/data4/lty/victory_data/testB_dataset_spacing662'
    model_path  = '/mnt/data4/lty/victory/train_dir/retinanet_resnet101_C2345_small_anchor_alpha8_spacing662_const_scale_data_test0.3718/resnet101_csv_18_test_0.3718.h5'
    result_path = '/mnt/data4/lty/victory/result_dir/retinanet_resnet101_C2345_small_anchor_alpha8_spacing662_const_scale_data_test0.3718/testB_resnet101_csv_18_nms_0.6_score_threshold0.2.csv'
    
    #加载模型
    print('model loading')
    backbone = 'resnet101'
    model = models.load_model(model_path, backbone_name=backbone)
    model = models.convert_model(model, nms=True, nms_threshold=0.6)
    print('model loaded')
    
    predictTest(model, TEST_DATA_DIR, result_path, score_threshold=0.2, input_min_max_side=(768, 768))
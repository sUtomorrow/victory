import os
import numpy as np
import cv2
import pickle
import pandas as pd

CLASS_ID_TO_MODEL_CLASS_IDX = {
    '1' : 0,
    '5' : 1,
    '31': 2,
    '32': 3
}

DATA_CSV_COLUMNS = ['data_path', 'x1', 'y1', 'x2', 'y2', 'label']

INPUT_SLICE = 3

def collect_annotations(data_dir):
    patient_slice_annotations = {}
    all_pkl_file_names = [file_name for file_name in os.listdir(data_dir) if '.pkl' in file_name]
    for pkl_file_name in all_pkl_file_names:
        seriesuid = pkl_file_name.split('.pkl')[0]
        patient_slice_annotations[seriesuid] = {}
        pkl_file_path = os.path.join(data_dir, pkl_file_name)
        with open(pkl_file_path, 'rb') as fp:
            pkl_file = pickle.load(fp)
        for annotation in pkl_file['annotations']: # [x, y, z, dx, dy, dz, label]
            x_start = annotation[0] - annotation[3] / 2
            x_end   = x_start + annotation[3]
            x_start = int(round(x_start))
            x_end   = max(int(round(x_end)), x_start + 1)

            y_start = annotation[1] - annotation[4] / 2
            y_end   = y_start + annotation[4]
            y_start = int(round(y_start))
            y_end   = max(int(round(y_end)), y_start + 1)

            z_start = annotation[2] - annotation[5] / 2
            z_end   = z_start + annotation[5]
            z_start = int(round(z_start))
            z_end   = max(int(round(z_end)), z_start + 1)
            for slice_idx in range(z_start, z_end):
                if slice_idx not in patient_slice_annotations[seriesuid]:
                    patient_slice_annotations[seriesuid][slice_idx] = []
                patient_slice_annotations[seriesuid][slice_idx].append([x_start, y_start, x_end, y_end, annotation[6]])
    return patient_slice_annotations


def crop_input_data(data, slice_idx):
    input_slice_satrt = slice_idx - INPUT_SLICE // 2
    input_slice_end   = input_slice_satrt + INPUT_SLICE
    padding_satrt = 0
    padding_end = 0
    if input_slice_satrt < 0:
        padding_satrt = -input_slice_satrt
        input_slice_satrt = 0
    if input_slice_end > data.shape[2]:
        padding_end = input_slice_end - data.shape[2]
        input_slice_end = data.shape[2]
    input_data = data[:, :, input_slice_satrt:input_slice_end]
    if padding_satrt or padding_end:
        input_data = np.pad(input_data, ((0, 0), (0, 0), (padding_satrt, padding_end)), mode='constant', constant_values=0)
    return input_data


def data_name_encode(seriesuid, slice_idx):
    return '%s_%d.npy' % (seriesuid, slice_idx)


def generate_class_csv(class_id_to_model_class_idx):
    class_csv = pd.DataFrame(columns=['class', 'id'])
    for key in class_id_to_model_class_idx.keys():
        class_csv = class_csv.append({
            'class': key,
            'id': class_id_to_model_class_idx[key]
        }, ignore_index = True)
    return class_csv


def data_prepare_for_retinanet(data_dir, target_dir, data_split_info_path, imshow=False):
    train_data_dir = os.path.join(target_dir, 'train')
    valid_data_dir = os.path.join(target_dir, 'valid')
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    if not os.path.exists(valid_data_dir):
        os.mkdir(valid_data_dir)
    
    train_data_csv_dict = {}
    valid_data_csv_dict = {}

    for column in DATA_CSV_COLUMNS:
        train_data_csv_dict[column] = []
        valid_data_csv_dict[column] = []
    
    print('collecting annotations...')
    patient_slice_annotations = collect_annotations(data_dir)
    print('annotations collected')

    print('loading data split info...')
    with open(data_split_info_path, 'rb') as fp:
        data_split_info = pickle.load(fp)
    print('data split info loaded')

    data_idx = 0
    for seriesuid in patient_slice_annotations.keys():
        data_npy_path = os.path.join(data_dir, seriesuid + '.npy')
        data = np.load(data_npy_path) # zyx
        data = np.transpose(data, [1, 2, 0]) # zyx => yxz
        
        print(data_idx, seriesuid)

        if seriesuid in data_split_info['valid']:
            data_csv_dict = valid_data_csv_dict
            data_save_dir = valid_data_dir
        else:
            data_csv_dict = train_data_csv_dict
            data_save_dir = train_data_dir
        
        for slice_idx in patient_slice_annotations[seriesuid].keys():
            if slice_idx < 0 or slice_idx >= data.shape[2]:
                print(seriesuid, 'annotation:', patient_slice_annotations[seriesuid][slice_idx], 'slice_idx:', slice_idx, 'out of bounding!!!')
                continue
            data_idx += 1
            input_data = crop_input_data(data, slice_idx)
            
            data_file_name      = data_name_encode(seriesuid, slice_idx)
            data_file_save_path = os.path.join(data_save_dir, data_file_name)
            
            if imshow:
                img_show = ((np.clip(input_data, -1000, 400) + 1000) / 1400 * 255).astype(np.uint8)
            for annotation in patient_slice_annotations[seriesuid][slice_idx]:
                data_csv_dict[DATA_CSV_COLUMNS[0]].append(data_file_save_path)
                data_csv_dict[DATA_CSV_COLUMNS[1]].append(annotation[0])
                data_csv_dict[DATA_CSV_COLUMNS[2]].append(annotation[1])
                data_csv_dict[DATA_CSV_COLUMNS[3]].append(annotation[2])
                data_csv_dict[DATA_CSV_COLUMNS[4]].append(annotation[3])
                data_csv_dict[DATA_CSV_COLUMNS[5]].append(annotation[4])
                if imshow:
                    cv2.rectangle(img_show, (annotation[0], annotation[1]), (annotation[2], annotation[3]), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            if imshow:
                cv2.imshow('image annotation', img_show)
                k = cv2.waitKey()
                if k == ord('q'):
                    exit()
                elif k == ord('c'):
                    imshow = False
            if not os.path.exists(data_file_save_path):
                np.save(data_file_save_path, input_data)

    valid_data_csv_df = pd.DataFrame(valid_data_csv_dict, columns=DATA_CSV_COLUMNS)
    valid_data_csv_df.to_csv(os.path.join(target_dir, 'valid_label.csv'), index=False, header=None)
    
    train_data_csv_df = pd.DataFrame(train_data_csv_dict, columns=DATA_CSV_COLUMNS)
    train_data_csv_df.to_csv(os.path.join(target_dir, 'train_label.csv'), index=False, header=None)

    class_csv = generate_class_csv(CLASS_ID_TO_MODEL_CLASS_IDX)
    class_csv.to_csv(os.path.join(target_dir, 'class.csv'), index=False, header=None)

if __name__ == '__main__':
    # data_dir             = '/mnt/data3/victory/rescale_z_train_dataset/'
    # target_dir           = '/mnt/data4/lty/victory_data/rescale_z_data_for_retinanet/'
    data_dir             = '/mnt/data4/lty/victory_data/train_dataset_spacing662/'
    target_dir           = '/mnt/data4/lty/victory_data/data_for_retinanet_spacing662/'
    data_split_info_path = '/mnt/data3/victory/data_split.pkl'
    imshow               = True
    data_prepare_for_retinanet(data_dir, target_dir, data_split_info_path, imshow)
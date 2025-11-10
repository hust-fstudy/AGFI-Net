import os
import os.path as osp
import math
import random
import shutil
import glob2
import numpy as np


def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)


def copy_file_to_dir(filepath_list, target_dir):
    if not osp.exists(target_dir):
        os.makedirs(target_dir)
    for file_path in filepath_list:
        if osp.isfile(file_path):
            shutil.copy2(file_path, target_dir)


def create_split(data_path, train_ratio):
    total_dir = osp.join(data_path, 'total')
    class_list = os.listdir(total_dir)
    for each_class in class_list:
        each_class_path = osp.join(total_dir, each_class)
        file_path_list = glob2.glob(osp.join(each_class_path, '*'))
        train_file_path_index = random.sample(range(0, len(file_path_list)),
                                              math.ceil(len(file_path_list) * train_ratio))
        test_file_path_index = list(set(range(0, len(file_path_list))) - set(train_file_path_index))
        train_file_path = [file_path_list[i] for i in train_file_path_index]
        test_file_path = [file_path_list[i] for i in test_file_path_index]
        copy_file_to_dir(train_file_path, osp.join(data_path, 'train', str(each_class)))
        copy_file_to_dir(test_file_path, osp.join(data_path, 'test', str(each_class)))


def remove_files_in_dir(directory_path):
    if not osp.isdir(directory_path):
        print(f"The provided path {directory_path} is not a valid directory.")
        return
    for filename in os.listdir(directory_path):
        file_path = osp.join(directory_path, filename)
        try:
            if osp.isfile(file_path) or osp.islink(file_path):
                os.unlink(file_path)
            elif osp.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def min_max_normalization(arr, mx, mi=0):
    arr = arr.astype(float)
    epsilon = 1e-8
    min_val = np.min(arr, axis=0)
    max_val = np.max(arr, axis=0)
    de = max_val - min_val
    if np.any(de < epsilon):
        if arr.ndim == 1:
            arr[:] = 0.5
        else:
            small_index = np.where(de < epsilon)[0]
            big_index = np.where(de >= epsilon)[0]
            arr[:, small_index] = 0.5
            arr[:, big_index] = (mx - mi) * ((arr[:, big_index] - min_val[big_index]) /
                                             (max_val[big_index] - min_val[big_index])) + mi
        normalize_data = arr
    else:
        normalize_data = (mx - mi) * ((arr - min_val) / (max_val - min_val)) + mi
    return normalize_data

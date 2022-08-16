import scipy.io as sio
import numpy as np
import os.path as osp
import os
import csv
import wfdb
import tqdm
import pandas as pd
import json
import pickle

from torch.utils.data import Dataset
from torchvision import datasets
import torch
from torch.nn import Parameter

label_dict = {'NORM': 1, 'AFIB': 2, '1AVB': 3, 'CLBBB': 4, 'CRBBB': 5, 'PAC': 6, 'VPC': 7, 'STD_': 8, 'STE_': 9}


def mat_loader(path):
    data = sio.loadmat(path)['ECG'][0][0][2]

    return data.astype(np.float32)  # 12 leads (12, 5000)


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    """

    def __call__(self, signal):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return torch.from_numpy(signal)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def split(reference_path, total_file_info):
    with open(reference_path, 'r') as f:
        records = [l.strip().split(",") for l in f][1:]
    record_dict = {}
    for record in records:
        record_dict[record[0]] = record
    df = pd.read_csv(total_file_info)
    with open("TRAIN_reference.csv", "w") as csvfile_train:
        with open('VAL_reference.csv', 'w') as csvfile_val:
            with open('TEST_reference.csv', 'w') as csvfile_test:
                writer_train = csv.writer(csvfile_train)
                writer_val = csv.writer(csvfile_val)
                writer_test = csv.writer(csvfile_test)

                writer_train.writerow(['Recording', 'First_label', 'Second_label', 'Third_label'])
                writer_val.writerow(['Recording', 'First_label', 'Second_label', 'Third_label'])
                writer_test.writerow(['Recording', 'First_label', 'Second_label', 'Third_label'])
                for i in range(1, df.shape[0]):
                    record = df.loc[i]
                    file_name = record['filename']
                    if record['strat_fold'] == 10:
                        writer_test.writerow(record_dict[file_name])
                    elif record['strat_fold'] == 9:
                        writer_val.writerow(record_dict[file_name])
                    else:
                        writer_train.writerow(record_dict[file_name])
    csvfile_train.close()
    csvfile_val.close()
    csvfile_test.close()


class multi_label_dataset(Dataset):
    def __init__(self, data_path, reference_path, transform=None, target_transform=None, loader=mat_loader):
        super(multi_label_dataset, self).__init__()
        self.data_files = []
        target_list = {}
        with open(reference_path, 'r') as f:
            records = [l.strip().split(",") for l in f][1:]
        #print(records)
        for i, record in enumerate(records):
            file_name = record[0]
            self.data_files.append(os.path.join(data_path, file_name + '.mat'))
            target_list[file_name] = record[1:]
        #print(self.data_files)
        #print(target_list)
        self.target_list = target_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_ = self.data_files[index]
        signal = self.loader(file_)
        signal = self.transform(signal)
        file_name = file_.split("/")[-1].split(".")[0]
        target = self.target_list[file_name]
        #print(target)
        target = self.target_transform(target)
        #print(signal.size())
        #print(target)
        return signal, target
        #file_name

if __name__ == '__main__':
    # split("/home/workspace/huichen/codes/CPSC/REFERENCE.csv","/home/workspace/huichen/data/ICBEB/icbeb_database.csv")
    x = multi_label_dataset("/home/workspace/huichen/data/challenge/TRAIN/", "/home/yuzhenqin/MVKT-ECG-QIN/datasets/TRAIN_reference.csv")

import sys
import os
import scipy.io as sio
sys.path.append("..")
from utils.imagepreprocess_FA_exp3 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def mat_loader(path):
    data = sio.loadmat(path)['ECG'][0][0][2]

    return data.astype(np.float32)  # 12 leads (12, 5000)

class multi_label_dataset(Dataset):
    def __init__(self, data_path, reference_path, transform=None, target_transform=None, loader=mat_loader):
        super(multi_label_dataset, self).__init__()
        self.data_files = []
        target_list = {}
        with open(reference_path, 'r') as f:
            records = [l.strip().split(",") for l in f][1:]
        for i, record in enumerate(records):
            file_name = record[0]
            self.data_files.append(os.path.join(data_path, file_name + '.mat'))
            target_list[file_name] = record[1:]

        self.target_list = target_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_ = self.data_files[index]
        signal = self.loader(file_)
        signal = self.transform(signal)[0:]
        file_name = file_.split("/")[-1].split(".")[0]
        target = self.target_list[file_name]
        target = self.target_transform(target)
        return signal, target
        #, file_name




class ICBEBInstanceSample(Dataset):
    """
    ICBEBInstance+Sample Dataset
    """
    def __init__(self, root, reference_path, train=True,
                 transform=None, target_transform=None,
                 loader=mat_loader, k=4096, mode='exact', is_sample=True, percent=1.0):
        super(ICBEBInstanceSample, self).__init__()
                         
                         
        self.data_files = []
        target_list = []
        with open(reference_path, 'r') as f:
            records = [l.strip().split(",") for l in f][1:]
        for i, record in enumerate(records):
            file_name = record[0]
            self.data_files.append(os.path.join(root, file_name + '.mat'))
            target_list.append(record[1:])
        self.target_list = target_list
        #print(self.data_files)
        #print(self.target_list)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader   
                         
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 9
        
        num_samples = len(self.data_files)
        #print(num_samples)
        label = self.target_list
        #print(label)
        self.cls_positive = [[] for i in range(num_classes)]
        #print(self.cls_positive.shape)
        for i in range(num_samples):
            self.cls_positive[int(label[i][0])-1].append(i)
        #print(self.cls_positive)
        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])
        #print(self.cls_negative)
        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        #print(self.cls_negative[0].shape)
        #print(self.cls_positive[0].shape)
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)
        #print(self.cls_negative[1].shape)
        #print(self.cls_positive[1].shape)
    def __getitem__(self, index):
        
        signal, target = self.data_files[index], self.target_list[index]
        signal = self.loader(signal)
        

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target_one_hot = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target[0]], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[int(target[0])-1]) else False
            neg_idx = np.random.choice(self.cls_negative[int(target[0])-1], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return signal, target_one_hot, index, sample_idx
    
    def __len__(self):
        return len(self.data_files)


def get_icbeb_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0, args=None):
    """
    icbeb 2018
    """
    
    train_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/TRAIN_reference.csv"
    val_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/VAL_reference.csv"
    test_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/TEST_reference.csv"
    train_transforms, val_transforms, test_transforms = preprocess_strategy_exp3(12)   #默认12导联
    train_target_transforms, val_target_transforms, test_target_transforms = target_strategy(9)

    train_set = ICBEBInstanceSample(root=args.data, reference_path=train_target_path,
                                     transform=train_transforms, target_transform=train_target_transforms,
                                     loader=mat_loader,
                                     train=True,
                                     k=k,
                                     mode=mode,
                                     is_sample=is_sample,
                                     percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_set = multi_label_dataset(args.data, val_target_path, val_transforms, val_target_transforms)
    test_set =multi_label_dataset(args.data, test_target_path, test_transforms, test_target_transforms)
    val_loader = DataLoader(val_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=int(num_workers/2))
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, val_loader, test_loader, n_data
if __name__ =='__main__':
    train_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/TRAIN_reference.csv"
    train_transforms, val_transforms, test_transforms = preprocess_strategy_exp3(12)   #默认12导联
    train_target_transforms, val_target_transforms, test_target_transforms = target_strategy(9)
    train_set = ICBEBInstanceSample(root="/home/huichen/data/challenge/TRAIN/", reference_path=train_target_path,
                                     transform=train_transforms, target_transform=train_target_transforms,
                                     loader=mat_loader,
                                     train=True,
                                     k=4096,
                                     mode='exact',
                                     is_sample=True,
                                     percent=1.0)
    train_loader = DataLoader(train_set,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    for idx, data in enumerate(train_loader):
        # measure data loading time
        if idx ==1 :
            input, target, index, contrast_idx = data
            print(input.shape)
            print(target.shape)
            print(index.shape)
            print(contrast_idx.shape)
            print(index[0])
            print(contrast_idx[0][0])
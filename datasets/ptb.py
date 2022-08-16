import numpy as np
from sqlalchemy import true
from datasets.dataset_ptbxl import PTBXLdataset
from torch.utils.data import DataLoader

class PTBXLInstanceSample(PTBXLdataset):
    """
    PTBXLInstance+Sample Dataset
    """
    def __init__(self, data_folder, task, train=True,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(data_folder=data_folder, task=task, train=train)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = self.n_classes
        if self.train:
            num_samples = len(self.X_train)
            label = self.y_train
        else:
            num_samples = len(self.X_test)
            label = self.y_test

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[int(label[i][0])-1].append(i)


        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img, target = self.X_train[index].T, self.y_train[index]
        else:
            img, target = self.X_test[index].T, self.y_test[index]
        
        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
     
            replace = True if self.k > len(self.cls_negative[int(target[0])-1]) else False
            neg_idx = np.random.choice(self.cls_negative[int(target[0])-1], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx



def get_ptb_dataloaders_sample(task, batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    ptb
    """
    data_folder = "/home/huichen/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"

    train_set = PTBXLInstanceSample(data_folder=data_folder,
                                    task=task,
                                    train=True,
                                    k=k,
                                    mode=mode,
                                    is_sample=is_sample,
                                    percent=percent)
    n_data = len(train_set)
    num_classes = train_set.n_classes
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = PTBXLdataset(data_folder=data_folder,
                          task=task,
                          train=False,test =True)
    print(len(test_set))
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=int(num_workers/2))
    val_dataset = PTBXLdataset(data_folder=data_folder,
                          task=task,
                          train=False, val=True)
    print(len(val_dataset))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=True,num_workers=num_workers)

    return train_loader, val_loader, test_loader, n_data, num_classes
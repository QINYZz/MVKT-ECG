from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch
import cv2
import random
import numpy as np
import pywt
import pdb


class denosing(object):
    """denosing the signal"""
    def __init__(self, num_leads):
        self.num_leads = num_leads
    @staticmethod
    def WTfilt_1d_db6(signal, num_leads):

        """
        # 
        # Martis R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
        #     wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
        :param sig:1-D numpy Array
        :return: 1-D numpy Array
        """
        de_sig = []
        for i in range(num_leads):
            signal_ = signal[i, :]
            coeffs = pywt.wavedec(signal_, 'db6', level=9)
            # print(coeffs)
            coeffs[-1] = np.zeros(len(coeffs[-1]))
            coeffs[-2] = np.zeros(len(coeffs[-2]))
            coeffs[0] = np.zeros(len(coeffs[0]))
            sig_filt = pywt.waverec(coeffs, 'db6')
            de_sig.append(sig_filt)
        de_sig = np.array(de_sig)
        return de_sig
        
        
    @staticmethod
    def WTfilt_1d_db4(signal, num_leads):
        """
        another filter method
        """

        de_sig = []
        for i in range(num_leads):
            signal_ = signal[i, :]
            
            coeffs = pywt.wavedec(signal_, 'db4', level=9)
            # print(coeffs)
            coeffs[-1] = np.zeros(len(coeffs[-1]))
            coeffs[-2] = np.zeros(len(coeffs[-2]))
            coeffs[0] = np.zeros(len(coeffs[0]))
            sig_filt = pywt.waverec(coeffs, 'db4')
            de_sig.append(sig_filt)

        de_sig = np.array(de_sig)
        return de_sig

    def __call__(self, signal):
        alpha = random.uniform(0, 1)
        if  alpha > 0.5:
            return self.WTfilt_1d_db6(signal, self.num_leads)
        else:
            return self.WTfilt_1d_db4(signal, self.num_leads)
            
class denosing_db6(object):
    """denosing the signal"""
    def __init__(self, num_leads):
        self.num_leads = num_leads
    @staticmethod
    def WTfilt_1d_db6(signal, num_leads):
        """
        # 
        # Martis R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
        #     wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
        :param sig:1-D numpy Array
        :return: 1-D numpy Array
        """
        de_sig = []
        for i in range(num_leads):
            signal_ = signal[i, :]
            coeffs = pywt.wavedec(signal_, 'db6', level=9)
            # print(coeffs)
            coeffs[-1] = np.zeros(len(coeffs[-1]))
            coeffs[-2] = np.zeros(len(coeffs[-2]))
            coeffs[0] = np.zeros(len(coeffs[0]))
            sig_filt = pywt.waverec(coeffs, 'db6')
            de_sig.append(sig_filt)
  
        de_sig = np.array(de_sig)
        return de_sig  
               
    def __call__(self, signal):
        return self.WTfilt_1d_db6(signal, self.num_leads)

            
class RandomCrop1D(object):
    """
    Unified length of the signal
    Args:
        signal: signals need to be unified
        target_length: unified length
    """
    def __init__(self, target_length):
        self.target_length = target_length
    @staticmethod
    def padCrop(signal, target_length):
        # import pdb
        # pdb.set_trace()
        ori_length = signal.shape[1]
        if ori_length >= target_length:
            target_loc_max = int(ori_length - target_length)
            target_loc = random.randint(0, target_loc_max)
            signal = signal[:, target_loc: target_loc + target_length]

        else:
            pad_signal = signal
            while (target_length - pad_signal.shape[1]) >= ori_length:
                pad_signal = np.concatenate((pad_signal, signal),axis=1)
            if target_length - pad_signal.shape[1] > 0:
                pad_num = target_length - pad_signal.shape[1]
                signal = np.concatenate((pad_signal, signal[:, :pad_num]), axis=1)
            else:
                signal = pad_signal

        return signal

    def __call__(self, signal):
        return self.padCrop(signal, self.target_length)

    def __repr__(self):
        return self.__class__.__name__ + '(target_length={0})'.format(self.target_length)



class CenterCrop1D(object):
    """
    Unified length of the signal
    Args:
        signal: signals need to be unified
        target_length: unified length
    """
    def __init__(self, target_length):
        self.target_length = target_length
    @staticmethod
    def padCrop(signal, target_length):
        # import pdb
        # pdb.set_trace()
        ori_length = signal.shape[1]
        if ori_length >= target_length:
            target_loc = int(ori_length - target_length) // 2
            signal = signal[:, target_loc: target_loc + target_length]
        else:
            pad_signal = signal
            while (target_length - pad_signal.shape[1]) >= ori_length:
                pad_signal = np.concatenate((pad_signal, signal),axis=1)
            if target_length - pad_signal.shape[1] > 0:
                pad_num = target_length - pad_signal.shape[1]
                signal = np.concatenate((pad_signal, signal[:, :pad_num]), axis=1)
            else:
                signal = pad_signal
        return signal

    def __call__(self, signal):
        return self.padCrop(signal, self.target_length)

    def __repr__(self):
        return self.__class__.__name__ + '(target_length={0})'.format(self.target_length)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    """"use mean&std to normalize"""
    @staticmethod
    def normalize(signal, mean, std):
        #print(signal.shape)
        for i in range(signal.shape[0]):
            signal[i, :] = (signal[i, :] - mean[i]) / std[i]
        return signal
    def __call__(self, signal):
        return self.normalize(signal, self.mean, self.std)
        


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
        return torch.from_numpy(signal).type(torch.FloatTensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class one_hot(object):
    """
    transform the label_list to one_hot
    """
    def __init__(self, num_classes, epsilon, train):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.train = train
    @staticmethod
    def _to_one_hot(label_list, epsilon, train, num_classes):
        """label_list:[2,3, '','','','','','','']"""
        _label = np.zeros([num_classes])
        for label in label_list:
            if label != '':
                _label[int(label)-1] = 1
        num = sum(_label)
        label = np.zeros([num_classes])
        # label smoothing
        for i in range(9):
            label[i] = (num - epsilon) / num if _label[i] == 1 else epsilon / (9 - num)
        if train:
            _label = label
        return _label

    def __call__(self, label_list):
        return self._to_one_hot(label_list, self.epsilon, self.train, self.num_classes)



class Flip(object):
    """"flip the signal[1, 10000]"""
    @staticmethod
    def flip(signal):
        signal = -signal
        return signal
    def __call__(self, signal):
        return self.flip(signal)
        
def preprocess_strategy_exp3(num_leads):
    train_transforms = transforms.Compose([
            denosing_db6(num_leads),
            CenterCrop1D(10000),
            #Flip(),
            #Normalize([0, 0, 0,0,0,0,-0.0001, -0.0002, 0, 0, 0, 0.0002],[0.1441, 0.1732, 0.1582, 0.1398, 0.1260, 0.1508, 0.3119, 0.4467, 0.4224, 0.4036, 0.3867, 0.3829]),
            ToTensor(),
            
            # normalize,
        ])
    val_transforms = transforms.Compose([
            denosing_db6(num_leads),
            CenterCrop1D(10000),
            #Normalize([0.0001,0, 0, -0.0002, 0, 0, -0.0002, -0.0004, 0, 0, 0.0003, 0.0003], [0.1332, 0.1641, 0.1461, 0.1310, 0.1147, 0.1409, 0.2655, 0.3770, 0.3689, 0.3376, 0.3424, 0.4300]),
            ToTensor(),
            
        ])
    test_transforms = transforms.Compose([
            denosing_db6(num_leads),
            CenterCrop1D(10000),
            #Normalize([-0.0002, -0.0002, 0, 0.0002, 0, -0.0001, 0.0004, 0.0005, 0.0005, 0.0003, -0.0002, -0.0002], [0.1441, 0.1740, 0.1569, 0.1407, 0.1247, 0.1506, 0.3074, 0.4433, 0.4164, 0.3722, 0.4110, 0.4041]),
            ToTensor(),
            
        ])
        
    return train_transforms, val_transforms, test_transforms


def target_strategy(num_classes):
    train_target_transforms = transforms.Compose([
            one_hot(num_classes, 0.1, train=False),
            ToTensor(),
        ])
    val_target_transforms = transforms.Compose([
            one_hot(num_classes, 0.1, train=False),
            ToTensor(),
        ])
    test_target_transforms = transforms.Compose([
            one_hot(num_classes, 0.1, train=False),
            ToTensor(),
        ])
    return train_target_transforms, val_target_transforms, test_target_transforms
    
if __name__ == '__main__':
    signal = np.ones(5000)
    signal_d = denosing()
    print(signal_d(signal).shape)
        

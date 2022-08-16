import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import random
import shutil
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pdb
from tqdm import tqdm

from utils.functions import *
from datasets.dataset_cpsc import multi_label_dataset
from datasets.dataset_ptbxl import PTBXLdataset
from datasets.ptb import get_ptb_dataloaders_sample

from utils.imagepreprocess_FA_exp3 import *
from utils.sampler import ImbalancedDatasetSampler
from models.CNN_Ag import cnn_Ag
from models.xresnet_1d import *
# from models.xresnet1d import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from fastai.torch_core import *
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
import csv

parser = argparse.ArgumentParser(description='PyTorch ECG Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--task', default='rhythm',
                    help='train task')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', default=[40, 70], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="1,2", type=str, 
                    help='GPU ids to use.')
parser.add_argument('--backbone', default=None, type=str,
                    help='name of backbone')
parser.add_argument('--modeldir', default=None, type=str,
                    help='director of checkpoint')
parser.add_argument('--num-leads', default=12, type=int,
                    help='define the number of leads total')
parser.add_argument('--freezed-layer', default=None, type=int,
                    help='define the end of freezed layer')
parser.add_argument('--store-model-everyepoch', dest='store_model_everyepoch', action='store_true',
                    help='store checkpoint in every epoch')
parser.add_argument('--classifier-factor', default=None, type=int,
                    help='define the multiply factor of classifier')
parser.add_argument('--benchmark', default=None, type=str,
                    help='name of dataset')
parser.add_argument('--test_per_epoch', dest='test_per_epoch', action='store_true',
                    help='test_per_epoch')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


best_prec1 = 0
min_loss = 1
best_val_auc = 0
best_test_auc = 0

def main():
    global args, best_prec1, min_loss, best_val_auc, best_test_auc
    args = parser.parse_args()

    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    # Data loading code
    # multi
    if args.benchmark == "PTBXL":
        train_dataset = PTBXLdataset(args.data, args.task, train=True)
        val_dataset = PTBXLdataset(args.data,args.task,train = False, val=True)
        test_dataset = PTBXLdataset(args.data, args.task, train = False,test=True)
        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))
        args.num_classes = train_dataset.n_classes
        print(args.num_classes)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    if args.benchmark == "CPSC":
        args.num_classes = 9
        train_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/TRAIN_reference.csv"
        val_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/VAL_reference.csv"
        test_target_path = "/home/yuzhenqin/MVKT-ECG-QIN/datasets/TEST_reference.csv"
        #print(train_target_path)
        train_transforms, val_transforms, test_transforms = preprocess_strategy_exp3(12)
        train_target_transforms, val_target_transforms, test_target_transforms = target_strategy(args.num_classes)
        # multi
        train_dataset = multi_label_dataset(args.data, train_target_path, train_transforms, train_target_transforms)
        #print(train_dataset.size())
        val_dataset = multi_label_dataset(args.data, val_target_path, val_transforms, val_target_transforms)
        test_dataset = multi_label_dataset(args.data, test_target_path, test_transforms, test_target_transforms)
        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))
        print(args.num_classes)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset), drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size = args.batch_size, shuffle=False,   #batch_size = args.batch_size
            num_workers=args.workers, pin_memory = True)
    # create model
  
    model = eval(args.backbone)(num_classes=args.num_classes, num_leads=args.num_leads)
    #apply_init(model.fc, nn.init.kaiming_normal_)
    print(model)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("msg.missing_keys:", msg.missing_keys)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    device_ids=range(torch.cuda.device_count())  #torch.cuda.device_count()=2
    if args.gpu is not None:
        model = model.cuda()
        if len(device_ids)>1:
            model=nn.DataParallel(model) 
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # model.features = torch.nn.DataParallel(model.features)
        model.cuda()

    # define loss function (criterion) and optimizer
    # multi_loss
    criterion_cls = nn.BCEWithLogitsLoss(reduce=True, size_average=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # plot network
    # vizNet(model, args.modeldir)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True




    ## init evaluation data loader
    evaluate_transforms = None
    if evaluate_transforms is not None:
        eval_dataset = multi_label_dataset(evaldir, eval_target_path, evaluate_transforms, test_target_transforms)
        evaluate_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        if evaluate_transforms is not None:
            results = torch.zeros(len(evaluate_loader), num_classes)
            print("=> start evaluation")
            model_file = os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (args.eval_epoch + 1))
            print("=> loading model '{}'".format(model_file))
            best_model = torch.load(model_file)
            model.load_state_dict(best_model['state_dict'])
            result, names = evaluate(evaluate_loader, model)
        return
    # make directory for storing checkpoint files
    args.modeldir = args.modeldir
    if os.path.exists(args.modeldir) is not True:
        os.mkdir(args.modeldir)
    stats_ = stats(args.modeldir, args.start_epoch)
    val_auc_list = []
    test_auc_list = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion_cls, optimizer, epoch, args)
        valObj, prec1, prec5 = validate(val_loader, model, criterion_cls, epoch, args)
        stats_._update(trainObj,top1, valObj,prec1)
        
        results,targets = evaluate(val_loader,model,args)
        val_auc = roc_auc_score(targets, results, average='macro')
        print('val_auc metric in {} epoch is {}'.format(epoch, val_auc))
        val_auc_list.append(val_auc) 

        is_best_val = val_auc > best_val_auc
        best_val_auc = max(val_auc,best_val_auc)

        #filename = os.path.join(args.modeldir, 'model_best.pth.tar')
        if args.test_per_epoch:
            results2, targets2 = evaluate(test_loader, model, args)
            test_auc = roc_auc_score(targets2, results2, average='macro')
            print('test_auc metric in {} epoch is {}'.format(epoch, test_auc))
            test_auc_list.append(test_auc)
            is_best_test = test_auc > best_test_auc
            best_test_auc = max(test_auc,best_test_auc)
        filename = []
        if args.store_model_everyepoch:
            filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
        else:
            filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best_test.pth.tar'))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_auc': best_val_auc,      #best_prec1没有来历
            'best_test_auc': best_test_auc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_val, is_best_test, filename)
        plot_curve(stats_, args.modeldir, True)
        data = stats_
        sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
        '''  if auc > best_auc:
                best_auc = auc
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer' : optimizer.state_dict() }, filename)'''
        
        '''if epoch == args.eval_epoch:
            model_file = filename[1]
            print("=> loading best model '{}'".format(model_file))
            print("=> start evaluation")
            best_model = torch.load(model_file)
            model.load_state_dict(best_model['state_dict'])
            result, names = evaluate(evaluate_loader, model)
            # calculate the accuracy on the test datatset
            measure(result)'''
    print('===================finish the training====================')
    print("最佳验证auc：",max(val_auc_list), val_auc_list.index(max(val_auc_list)))
    print("最佳测试auc：",max(test_auc_list), test_auc_list.index(max(test_auc_list)))


def train(train_loader, model, criterion_cls, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        signal = input        
        if args.gpu is not None:
            signal = signal.cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True)
        target = target.float() # multi
        if args.num_leads == 1:
            signal = signal[:,[0],:]
        #print("信号的形状：", signal.shape)
        output = model(signal)
        #print("输出的形状：", output.shape)
        loss = criterion_cls(output, target)
        # measure accuracy and record loss
        # multi_label
        prec1, prec5 = multi_accuracy(torch.sigmoid(output), target, epoch=epoch, topk=(1, 1),)

        losses.update(loss.item(), signal.size(0))
        top1.update(prec1[0], signal.size(0))
        top5.update(prec5[0], signal.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        ### avoid NAN
        # nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion_cls, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to validate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            signal = input
            if args.gpu is not None:
                signal = signal.cuda(non_blocking=True).float()
            
            target = target.cuda(non_blocking=True)
            target = target.float() # multi
            if args.num_leads == 1:
                signal = signal[:,[0],:]
            output = model(signal)
            loss = criterion_cls(output, target)

            # measure accuracy and record loss

            # multi_label
            prec1, prec5 = multi_accuracy(torch.sigmoid(output), target, epoch=1, topk=(1, 1),)
            losses.update(loss.item(), signal.size(0))
            
            top1.update(prec1[0], signal.size(0))  # multi-label:item()
            top5.update(prec5[0], signal.size(0))  # multi-label:item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top1.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def evaluate(test_loader, model, args):
    # switch to evaluate mode
    model.eval()
    results = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm(enumerate(test_loader)):
            signal = input
            if args.gpu is not None:
                signal = signal.cuda(non_blocking=True).float()
            if args.num_leads == 1:
                signal = signal[:,[0],:]
            output = model(x=signal)
            output = torch.sigmoid(output)
            results.append(output.cpu())
            targets.append(target.cpu())
    results = torch.cat(results)
    targets = torch.cat(targets)
    return results, targets


'''def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])'''
def save_checkpoint(state, is_val_best, is_best_test, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_val_best:
        shutil.copyfile(filename[0], filename[1])
    if is_best_test:
        shutil.copyfile(filename[0], filename[2])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Learning_rate_generater(object):
    """Generates a list of learning rate for each training epoch"""
    def __init__(self, method, params, total_epoch):
        if method == 'step':
            lr_factor, lr = self.step(params, total_epoch)
        elif method == 'log':
            lr_factor, lr = self.log(params, total_epoch)
        else:
            raise KeyError("=> undefined learning rate method '{}'" .format(method))
        self.lr_factor = lr_factor
        self.lr = lr
    def step(self, params, total_epoch):
        decrease_until = params[0]
        decrease_num = len(decrease_until)
        base_factor = 0.1
        lr_factor = torch.ones(total_epoch, dtype=torch.double)
        lr = [args.lr]
        for num in range(decrease_num):
            if decrease_until[num] < total_epoch:
                lr_factor[int(decrease_until[num])] = base_factor
        for epoch in range(1,total_epoch):
            lr.append(lr[-1]*lr_factor[epoch])
        return lr_factor, lr
    def log(self, params, total_epoch):
        params = params[0]
        left_range = params[0]
        right_range = params[1]
        np_lr = np.logspace(left_range, right_range, total_epoch)
        lr_factor = [1]
        lr = [np_lr[0]]
        for epoch in range(1, total_epoch):
            lr.append(np_lr[epoch])
            lr_factor.append(np_lr[epoch]/np_lr[epoch-1])
        if lr[0] != args.lr:
            args.lr = lr[0]
        return lr_factor, lr


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    if epoch < 15:
        lr = 0.001 * (epoch + 1)
    else:
        lr = args.lr
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def multi_accuracy(output, target, epoch, topk=(1,)):
    """computs the numti-label accuracy"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # get set of labels 
        predict = torch.where(output > 0.5, torch.full_like(output, 1), torch.full_like(output, 0))
        
        # label_dict = {0:'AF', 1:'I-AVB', 2:'LBBB', 3:'RBBB', 4:'PAC', 5:'PVC', 6:'PAC', 7:'STD', 8:'STE'}
        TP = torch.zeros(args.num_classes)
        FP = torch.zeros(args.num_classes)
        TN = torch.zeros(args.num_classes)
        FN = torch.zeros(args.num_classes)
        #pdb.set_trace()
        Accuracy = torch.zeros(args.num_classes)
        Precision = torch.zeros(args.num_classes) 
        Recall =  torch.zeros(args.num_classes)
        F1 =  torch.zeros(args.num_classes)
        for label in range(args.num_classes):
            for sample in range(batch_size):
                if predict[sample, label] == target[sample, label] == 1:
                    TP[label] += 1
                if predict[sample, label] == 1 and target[sample, label] == 0:
                    FP[label] += 1
                if predict[sample, label] ==0 and target[sample, label] == 1:
                    FN[label] += 1
                if predict[sample, label] == target[sample, label] == 0:
                    TN[label] += 1
        for i in range(args.num_classes):
            Accuracy[i] = (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i]) 
            Precision[i] = TP[i] / (TP[i] + FP[i]) 
            Recall[i] = TP[i] / (TP[i] + FN[i]) 
            F1[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])    # 2*(Precision*Recall)/(Precision+Recall)
        Accuracy_Macro = Accuracy.float().sum(0, keepdim=True).mul_(1.0 / args.num_classes)
        Precision_Macro = Precision.float().sum(0, keepdim=True).mul_(1.0 / args.num_classes)
        Recall_Macro = Recall.float().sum(0, keepdim=True).mul_(1.0 / args.num_classes)
        F1_Macro = F1.float().sum(0, keepdim=True).mul_(1.0 / args.num_classes)
    
        
        if epoch % 10 == 0:
            print ('Accuracy_Macro, Precision_Macro, Recall_Macro, F1_Macro', Accuracy_Macro, Precision_Macro, Recall_Macro, F1_Macro)
        res = []
        for k in topk:
            res.append(Accuracy_Macro)
        return  res
        
    

if __name__ == '__main__':
    main()


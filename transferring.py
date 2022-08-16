import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] ="0,4"
import random
import shutil
import time
from unicodedata import name
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pdb
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from utils.functions import *
import torch.utils.data
import torch.utils.data.distributed
from datasets.ptb import get_ptb_dataloaders_sample
from datasets.icbeb import get_icbeb_dataloaders_sample
from models.xresnet_1d import *
from tqdm import tqdm
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from loss.criterion import CRDLoss
# from models.xresnet1d import *


parser = argparse.ArgumentParser(description='Single_finetune')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--task', default='subdiagnostic',
                    help='train task')
parser.add_argument('--student_backbone', default='resnet50',
                    help='student architecture: ')
parser.add_argument('--teacher_backbone', default='resnet50',
                    help='teacher architecture: ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
# CRD distillation
parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

# KL distillation
parser.add_argument('--kd_T', type=float, default=1,  help='temperature for KD distillation')

# NCE distillation
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=1024, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--schedule', default=[40, 70], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
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
parser.add_argument('--eval-epoch', default=119, type=int,
                    help='name of dataset')
parser.add_argument('--test_per_epoch', dest='test_per_epoch', action='store_true',
                    help='test_per_epoch')     
parser.add_argument('--pretrained', metavar='pretrained ckpt',
                    help='path to pretrained ckpt of model_teacher')          
parser.add_argument('--student_pretrain', metavar='pretrained ckpt',
                    help='path to pretrained ckpt of model_student')                        

best_val_auc = 0
best_test_auc = 0             
best_prec1 = 0
min_loss = 1
best_auc = 0
def main():
    global args, best_prec1, min_loss, best_auc, best_val_auc, best_test_auc
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
    if args.benchmark == "PTBXL":
        train_loader, val_loader, test_loader, n_data, num_classes = get_ptb_dataloaders_sample(task=args.task,
                                                                    batch_size=args.batch_size,
                                                                    num_workers=args.workers,
                                                                    k=args.nce_k,
                                                                    mode=args.mode)
        args.num_classes = num_classes
        print(num_classes)
    if args.benchmark == "CPSC":
        train_loader, val_loader, test_loader, n_data = get_icbeb_dataloaders_sample(batch_size=args.batch_size,
                                                                        num_workers=args.workers,
                                                                        k=args.nce_k,
                                                                        mode=args.mode,
                                                                        args=args)
        args.num_classes = 9
        print(args.num_classes)
    # create model
    model_t = eval(args.teacher_backbone)(num_classes=args.num_classes)
    model_s = eval(args.student_backbone)(num_classes=args.num_classes, num_leads=1)
    
    '''if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            #state_dict = checkpoint['model']
            state_dict = checkpoint['state_dict']
            print("check teacher checkpoint")
            """
            # get weight in the final fc layer to init the student model
            state_dict_4_student = dict()
            for key, value in state_dict.items():
                if key.startswith("fc"):
                    state_dict_4_student[key] = value
            model_s.load_state_dict(state_dict_4_student, strict=False)
            """
            #model_t.load_state_dict(state_dict, strict=False)
            # 
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))'''
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            print("check teacher checkpoint")
            #from IPython import embed
            #embed()
            model_t.load_state_dict(state_dict, strict=False) #加载老师模型
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
   
    if args.student_pretrain:
        if os.path.isfile(args.student_pretrain):
            print("=> loading checkpoint for student'{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            student_state_dict = checkpoint['state_dict']
            if "moco_new" in args.student_pretrain:
                for k in list(student_state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        student_state_dict[k[len("module.encoder_q."):]] = student_state_dict[k]
                # delete renamed or unused k
                del student_state_dict[k]
            else:
                for k in list(student_state_dict.keys()):
                    # remove prefix, and filter fc parameter
                    if not k.startswith("module.fc"):
                        student_state_dict[k[len("module."):]] = student_state_dict[k]
            print("check student checkpoint")
            from IPython import embed
            embed()
            model_s.load_state_dict(student_state_dict, strict=False)
            print("=> loaded pre-trained model for student'{}'".format(args.student_pretrain))
        else:
            print("=> no checkpoint found at '{}'".format(args.student_pretrain))
                
            
            
            
    data = torch.randn(2, 12, 10000)
    model_t.eval()
    model_s.eval()
    _, feat_t, _ = model_t(data, is_feat_crd=True)
    _, feat_s, _ = model_s(data[:, [0], :], is_feat_crd=True)
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    device_ids=range(torch.cuda.device_count())  #torch.cuda.device_count()=2

    if args.gpu is not None:
        model_s = model_s.cuda()
        model_t = model_t.cuda()
        if len(device_ids)>1:
            model_s=nn.DataParallel(model_s) #???model??.cuda() ?
            model_t=nn.DataParallel(model_t)
    elif args.distributed:
        model_s.cuda()
        model_s = torch.nn.parallel.DistributedDataParallel(model_s)
        model_t.cuda()
        model_t = torch.nn.parallel.DistributedDataParallel(model_t)
    else:
        # model.features = torch.nn.DataParallel(model.features)
        model_s.cuda()
        model_t.cuda()
        
    #测试一下教师的性能
    resultss,targetss = evaluate2(test_loader,model_t)
    teacher_auc = roc_auc_score(targetss, resultss, average='macro')
    print('教师的性能：',teacher_auc)

    # define loss function (criterion) and optimizer
    # multi_loss
    criterion_cls = nn.BCEWithLogitsLoss(reduce=True, size_average=True).cuda()
    criterion_kl = nn.KLDivLoss().cuda()
    criterion_dist = nn.MSELoss(size_average=True).cuda()
    args.s_dim = feat_s[-1].shape[1]
    args.t_dim = feat_t[-1].shape[1]
    args.n_data = n_data   #TODO
    criterion_kd = CRDLoss(args)
    module_list.append(criterion_kd.embed_s)
    module_list.append(criterion_kd.embed_t)
    trainable_list.append(criterion_kd.embed_s)
    trainable_list.append(criterion_kd.embed_t)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    criterion_list.append(criterion_dist)
    
    
    optimizer = optim.Adam(trainable_list.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    # append teacher after optimizer to avoid weight_decay
    #optimizer = torch.optim.SGD(trainable_list.parameters(), args.lr,
     #                           momentum=args.momentum,
      #                          weight_decay=args.weight_decay)
    module_list.append(model_t)
    
    
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True


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
        
        # train for one epoch
        adjust_learning_rate(optimizer, epoch, args)
        trainObj, top1, top5 = train(epoch, train_loader, module_list, criterion_list, optimizer, args)
        valObj, prec1, prec5 = validate(val_loader, model_s, criterion_cls, epoch)
        stats_._update(trainObj,top1, valObj,prec1)
        

        results,targets = evaluate(val_loader,model_s)
        val_auc = roc_auc_score(targets, results, average='macro')
        print('val_auc metric in {} epoch is {}'.format(epoch, val_auc))
        val_auc_list.append(val_auc) 
        is_best_val = val_auc > best_val_auc
        best_val_auc = max(val_auc,best_val_auc)
        #filename = os.path.join(args.modeldir, 'model_best.pth.tar')
        if args.test_per_epoch:
            results2, targets2 = evaluate(test_loader, model_s)
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
            'state_dict': model_s.state_dict(),
            'best_prec1': best_prec1,      #best_prec1没有来历
            'best_test_auc': best_test_auc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_val, is_best_test, filename)
        plot_curve(stats_, args.modeldir, True)
        data = stats_
        sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
        '''results, targets = evaluate(test_loader, model_s)
            auc = roc_auc_score(targets, results, average='macro')
            print('auc metric in {} epoch is {}'.format(epoch, auc))
            auc_list.append(auc)
            if auc > best_auc:
                best_auc = auc
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model_s.state_dict(),
                'best_auc': best_auc,
                'optimizer' : optimizer.state_dict() }, filename)'''
        '''else:
            valObj, prec1, prec5 = validate(val_loader, model_t, model_s, criterion_kl, criterion_cls, epoch)
  
            # update stats
            stats_._update(trainObj, top1,  valObj, prec1)
            is_best = valObj < min_loss
            min_loss = min(valObj, min_loss)
            filename = []
            if args.store_model_everyepoch:
                filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
            else:
                filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
            filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename)
            plot_curve(stats_, args.modeldir, True)
            data = stats_
            sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})'''
    print('===================finish the training====================')
    print("最佳验证auc：",max(val_auc_list), val_auc_list.index(max(val_auc_list)))
    print("最佳测试auc：",max(test_auc_list), test_auc_list.index(max(test_auc_list)))
        
def kd_ce_loss(logits_S, logits_T, temperature):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def multi_label_KL_loss(logits_S, logits_T, temperature):
    logits_S = logits_S.sigmoid().unsqueeze(2)
    logits_T = logits_T.sigmoid().unsqueeze(2)
    
    logits_S = torch.cat([logits_S, 1-logits_S], dim=2)
    logits_T = torch.cat([logits_T, 1-logits_T], dim=2)
    ans = 0
    for i in range(args.num_classes):
        logits_S_i = logits_S[:, i, :]
        logits_T_i = logits_T[:, i, :]
        ans += nn.KLDivLoss()(torch.log(logits_S_i / temperature + 1e-8), logits_T_i / temperature + 1e-8)
    return ans
    
def multi_label_KL_loss_v2(logits_S, logits_T, temperature):
    logits_S = logits_S.sigmoid() / temperature
    logits_T = logits_T.sigmoid() / temperature
    
    loss = -logits_T * torch.log(logits_T) + logits_T * torch.log(logits_S) - (1-logits_T) * torch.log(1-logits_T) + (1-logits_T) * torch.log(1-logits_S)
    return torch.sum(loss) / logits_S.shape[0]
    

class RKD_loss(nn.Module): 
    """
    CVPR 2019 | Relational Knowledge Distillation
    """
    def __init__(self):
        super(RKD_loss, self).__init__()
        self.eps = 1e-12
        self.squared = False
    def pdist(self, e):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=self.eps)
    
        if not self.squared:
            res = res.sqrt()
    
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
    
        return res   
    def forward(self, student, teacher):
        t_d = self.pdist(teacher)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td
        
        s_d = self.pdist(student)
        mean_sd = s_d[s_d > 0].mean()
        s_d = s_d / mean_sd
        return F.smooth_l1_loss(s_d, t_d)
        
def train(epoch, train_loader, module_list, criterion_list, optimizer, args):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    
    criterion_cls = criterion_list[0]
    criterion_crd = criterion_list[1]
    criterion_dist = criterion_list[-1]
    model_s = module_list[0]
    model_t = module_list[-1]
    
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_kl = AverageMeter()
    losses_cls = AverageMeter()
    losses_crd = AverageMeter()
    losses_dist = AverageMeter()
    losses = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        # measure data loading time
        input, target, index, contrast_idx = data
        data_time.update(time.time() - end)
        signal = input.float()    
        if torch.cuda.is_available():
            signal = signal.cuda()
            target = target.cuda()
            index = index.cuda()
            
        target = target.float() # multi
        contrast_idx = contrast_idx.cuda()
        feats_s, feat_s, logit_s = model_s(signal[:, [0], :], is_feat_crd=True)
        with torch.no_grad():
            feats_t, feat_t, logit_t = model_t(signal, is_feat_crd=True)
            feat_t = [f.detach() for f in feat_t]
        # cls + kl div
        '''if epoch == 6:
            print("输入：",input)
            print("教师输出：",logit_t)
            print("学生输出：",logit_s)
            print("标签：",target)'''
        loss_cls = criterion_cls(logit_s, target)
        loss_kl = multi_label_KL_loss(logit_s, logit_t, args.kd_T)
        # CRD
        f_s = feat_s[-1]
        f_t = feat_t[-1]
        #print(contrast_idx.shape)
        loss_crd = criterion_crd(f_s, f_t, index, contrast_idx)

        '''if epoch > 20:
            loss = args.gamma * loss_cls + args.alpha * loss_kl + args.beta * loss_crd 
        else:
            loss = args.gamma * loss_cls + 0 * loss_kl + args.beta * loss_crd'''
        loss = args.gamma * loss_cls + args.alpha * loss_kl + args.beta * loss_crd
        # measure accuracy and record loss
        # multi_label
        prec1, prec5 = multi_accuracy(logit_s, target, epoch=epoch, topk=(1, 1),)

        losses.update(loss.item(), signal.size(0))
        
        losses_kl.update(loss_kl.item(), signal.size(0))
        losses_cls.update(loss_cls.item(), signal.size(0))
        losses_crd.update(loss_crd.item(), signal.size(0))
        #losses_dist.update(loss_dist.item(),signal.size(0))

        top1.update(prec1[0], signal.size(0))
        top5.update(prec5[0], signal.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # 'Loss_dist {loss_dist.val:.4f} ({loss_dist.avg:.4f})\t'
        # loss_dist=losses_dist,
        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_kl {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                  'Loss_crd {loss_crd.val:.4f} ({loss_crd.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_kl=losses_kl, loss_cls=losses_cls, loss_crd=losses_crd, loss=losses,
                   top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg
def validate(val_loader, model, criterion_cls, epoch):
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

'''def validate(val_loader, model, criterion_rkd,criterion_kl, criterion_cls, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_kl = AverageMeter()
    losses_cls = AverageMeter()
    losses = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to validate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, names) in enumerate(val_loader):
            signal = input
            if args.gpu is not None:
                signal = signal.cuda(non_blocking=True).float()
            
            target = target.cuda(non_blocking=True)
            target = target.float() # multi
            
            x_single = signal[:, 0, :].view([signal.shape[0], 1, signal.shape[-1]])
            output_teacher = model_t(signal)
            output_student = model_s(x_single)
            loss_kl = kd_ce_loss(output_student, output_teacher, args.kd_T).cuda()
            loss_cls = criterion_cls(output_student, target)  
            
            loss = loss_kl + loss_cls
            # measure accuracy and record loss

            # multi_label
            prec1, prec5 = multi_accuracy(torch.sigmoid(output), target, epoch=1, topk=(1, 1),)
            losses.update(loss.item(), signal.size(0))
           
            losses_kl.update(loss_kl.item(), signal.size(0))
            losses_cls.update(loss_cls.item(), signal.size(0))
            
            top1.update(prec1[0], signal.size(0))  # multi-label:item()
            top5.update(prec5[0], signal.size(0))  # multi-label:item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss_kl {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t'
                      'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top1.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss_kl=losses_kl, loss_cls=losses_cls, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg'''

def evaluate(test_loader, model):
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
            signal = signal[:,[0],:]
            output = model(x=signal)
            output = torch.sigmoid(output)
            results.append(output.cpu())
            targets.append(target.cpu())
    results = torch.cat(results)
    targets = torch.cat(targets)
    return results, targets  
def evaluate2(test_loader, model):
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
            output = model(x=signal)
            output = torch.sigmoid(output)
            results.append(output.cpu())
            targets.append(target.cpu())
    results = torch.cat(results)
    targets = torch.cat(targets)
    return results, targets  
'''def evaluate(test_loader, model):
    # switch to evaluate mode
    model.eval()
    result = torch.zeros(len(test_loader), args.num_classes)
    targets = torch.zeros(len(test_loader), args.num_classes)
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm(enumerate(test_loader)):
            signal = input
            if args.gpu is not None:
                signal = signal.cuda(non_blocking=True).float()
            
            x_single = signal[:, 0, :].view([signal.shape[0], 1, signal.shape[-1]])
            output = model(x=x_single) 
            output = torch.sigmoid(output)
            result[i, :] = output.cpu()
            targets[i, :] = target.cpu()
    return result, targets'''
    

def save_checkpoint(state, is_val_best, is_best_test, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_val_best:
        shutil.copyfile(filename[0], filename[1])
    if is_best_test:
        shutil.copyfile(filename[0], filename[2])
'''def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])'''


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
        lr = 0.0001 * (epoch + 1)
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
        
        TP = torch.zeros(args.num_classes)
        FP = torch.zeros(args.num_classes)
        TN = torch.zeros(args.num_classes)
        FN = torch.zeros(args.num_classes)
        #pdb.set_trace()
        Accuracy = torch.zeros(args.num_classes)
        Precision = torch.zeros(args.num_classes) 
        Recall =  torch.zeros(args.num_classes)
        F1 =  torch.zeros(args.num_classes)
        for label in range(int(args.num_classes)):
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


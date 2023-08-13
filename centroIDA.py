"""
@author: Ying Jin
@contact: sherryying003@gmail.com
"""
import math
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
# from examples.domain_adaptation.image_classification import mmd
from sampling import ClassAwareSampler
from tllib.self_training.mcc import MinimumClassConfusionLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.modules.entropy import entropy
from imblanced_data import imbalance_process, Reweight

from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.bsp import BatchSpectralPenalizationLoss


from functools import reduce
def add(x, y): return x + y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##class-wise feature alignment
def cluster_loss(feature_s,y_s, feature_t,y_t,class_centerall_t):
    batch_size, num_classes = y_s.size()

    A = (feature_s ** 2).sum(1).unsqueeze(dim=1)  # batch_size*1
    B = (feature_t ** 2).sum(1).unsqueeze(dim=1).t()  # 1*batch_size
    #The distance between any two examples
    distance_instance = (5e-4 + A + B - 2 * torch.matmul(feature_s, feature_t.t())).sqrt() ## batch_size*batch_size
    C = (class_centerall_t ** 2).sum(1).unsqueeze(dim=1).t()  # 1*num_classes
    #The distance between target examples and target class-centroids
    distance_target_center = (5e-4 + B.t() + C - 2 * torch.matmul(feature_t, class_centerall_t.t())).sqrt()# batch_size * num_classes

    batch_weight_s = F.softmax(y_s / 2, dim=1)  # batch_size*num_classes
    batch_weight_t = F.softmax(y_t / 2, dim=1)  # batch_size*num_classes
    predict_label_s = torch.max(batch_weight_s.detach(), dim=1)[1].unsqueeze(1)  # batch_size*1
    #Correct target pseudo labels based on nearset distance to target class-centroids
    predict_label_t = torch.min(distance_target_center.detach(), dim=1)[1].unsqueeze(1)
    # predict_label_t = torch.max(batch_weight_t.detach(), dim=1)[1].unsqueeze(1)  # batch_size*1
    predict_weight_s = batch_weight_s.gather(1, predict_label_s)  # batch_size*1
    predict_weight_t = batch_weight_t.gather(1, predict_label_t)  # batch_size*1
    # predict_weight_t_max = batch_weight_t.gather(1, predict_label_t_max)

    entropy_weight_s = entropy(batch_weight_s).detach()
    entropy_weight_s = 1 + torch.exp(-entropy_weight_s)
    entropy_weight_s = (batch_size * entropy_weight_s / torch.sum(entropy_weight_s)).unsqueeze(dim=1)  # batch_size x 1
    predict_weight_s = predict_weight_s * entropy_weight_s

    entropy_weight_t = entropy(batch_weight_t).detach()
    entropy_weight_t = 1 + torch.exp(-entropy_weight_t)
    entropy_weight_t = (batch_size * entropy_weight_t / torch.sum(entropy_weight_t)).unsqueeze(dim=1)  # batch_size x 1
    predict_weight_t = predict_weight_t * entropy_weight_t

    predict_weight_1_s = torch.zeros_like(batch_weight_s).to(device)
    predict_weight_1_s_index = predict_weight_1_s.scatter_(1, predict_label_s.detach(),1)  # batch_size*num_classes
    predict_weight_1_s = predict_weight_1_s_index * predict_weight_s
    predict_weight_1_t = torch.zeros_like(batch_weight_t).to(device)
    predict_weight_1_t_index = predict_weight_1_t.scatter_(1, predict_label_t.detach(),1)  # batch_size*num_classes
    predict_weight_1_t = predict_weight_1_t_index * predict_weight_t

    #The weight of same label between source and target
    same_class_weight = torch.matmul(predict_weight_1_s, predict_weight_1_t.t()).sqrt()
    same_class_index = torch.matmul(predict_weight_1_s_index, predict_weight_1_t_index.t())
    all_class_weight = (predict_weight_s * predict_weight_t.t()).sqrt()

    class_inner_distance = (distance_instance * same_class_weight.detach()).sum() / same_class_weight.detach().sum()
    class_dis_diatance = (distance_instance*all_class_weight.detach() - distance_instance*same_class_weight.detach()).sum() \
                         / (all_class_weight.detach().sum() - same_class_weight.detach().sum())


    loss = class_inner_distance / class_dis_diatance #* math.log(num_classes)


    return loss




# accumulative class-centroids
def class_center(classCenter_all, class_all_weight, feature, output, device):
    # _, feature_dim = classCenter_all.size()
    batch_weight = F.softmax(output/2, dim=1)    # batch_size*num_classes

    batch_size, num_classes = batch_weight.size()
    predict_label = torch.max(batch_weight.detach(), dim=1)[1].unsqueeze(1)  # batch_size
    predict_weight = batch_weight.gather(1, predict_label)   # batch_size*1

    predict_weight_1 = torch.zeros_like(batch_weight).to(device)
    predict_weight_1 = predict_weight_1.scatter_(1, predict_label.detach(),1)  # batch_size*num_classes
    predict_weight_1 = predict_weight_1 * predict_weight   # batch_size*num_classes

    yy = torch.ones(batch_size,1).to(device)
    class_batch_weight = torch.matmul(predict_weight_1.detach().t(), yy)

    class_centor_batch_all = torch.matmul(predict_weight_1.t(), feature)
    classCenter_all = (class_centor_batch_all + classCenter_all * class_all_weight)/(class_batch_weight + class_all_weight + 1e-5)
    class_all_weight += class_batch_weight

    return  classCenter_all, class_all_weight

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    def forward(self,class_centor_s,class_centor_t):
        num_classes ,_ = class_centor_s.size()
        A = (class_centor_s**2).sum(1).unsqueeze(dim=1)
        B = (class_centor_t**2).sum(1).unsqueeze(dim=1).t()
        class_num, _ =class_centor_t.size()

        st_distance = (5e-4 + A + B - 2 * torch.matmul(class_centor_s, class_centor_t.t()) ).sqrt()
        st_distance_loss = (class_num)*torch.diag(st_distance).sum()/(st_distance.detach().sum())

        loss = st_distance_loss #* math.log(num_classes)

        return loss



def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
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

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)


    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    #######################################################################################################

    print(dir(train_source_dataset))

    if args.data == 'OfficeHome':
        class_max_num =10000
        ratio_1 = 0.05
        ratio_2 = 0.05

        class_label = list(range(0,num_classes))
        train_source_dataset = imbalance_process(train_source_dataset,args.source[0],class_max_num,ratio_1,class_label)
        train_target_dataset = imbalance_process(train_target_dataset,args.target[0],class_max_num,ratio_2,class_label)


    num_class_s = np.unique(train_source_dataset.targets, return_counts=True)[1]
    print( num_class_s)
    num_class_t = np.unique(train_target_dataset.targets, return_counts=True)[1]
    print(num_class_t)


    num_class_tt = np.unique(test_dataset.targets, return_counts=True)[1]
    print(num_class_tt)
    s_max = torch.max(torch.tensor(num_class_s)) * num_classes
    t_max = torch.max(torch.tensor(num_class_t)) * num_classes
    instance_num = s_max if s_max>t_max else t_max

    ###########################################################################################################


    # print(train_source_dataset.targets)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     sampler=ClassAwareSampler(source_label=train_source_dataset.targets, batchsize=args.batch_size),
                                     shuffle=False, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)

    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters()+ domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    my_loss = Loss()

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args,num_classes, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier,optimizer,my_loss,
              lr_scheduler, epoch, args,num_classes,instance_num)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args,num_classes, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, num_classes,device)
    print("test_acc1 = {:3.1f}".format(acc1))

    # # decoupling, freezn backbone, resampling for a unbiased classifiser
    # train_source_loader_1 = DataLoader(train_source_dataset,
    #                                    sampler=ClassAwareSampler(source_label=train_source_dataset.targets,
    #                                                              batchsize=args.batch_size),
    #                                    batch_size=args.batch_size,
    #                                    shuffle=False, num_workers=args.workers, drop_last=True)
    # train_source_iter_1 = ForeverDataIterator(train_source_loader_1)
    # lr = 0.001
    # optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
    #                 lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # lr_scheduler = LambdaLR(optimizer, lambda x: lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    #
    # def freeze_backbone(self):
    #     print("Freezing backbone .......")
    #     for p in self.parameters():
    #         p.requires_grad = False
    #
    # freeze_backbone(classifier.backbone)
    # for epoch in range(1):
    #     print("lr:", lr_scheduler.get_lr())
    #     # pretrain for one epoch
    #     utils.empirical_risk_minimization(train_source_iter_1, classifier, optimizer,
    #                                       lr_scheduler, epoch, args,
    #                                       device)
    #     # validate to show pretrain process
    #     acc1 = utils.validate(val_loader, classifier, args, num_classes,device)
    #
    # acc1 = utils.validate(test_loader, classifier, args, num_classes, device)
    # print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier,  optimizer: SGD,my_loss:Loss,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace,num_classes,instance_num):
    batch_time = AverageMeter('Time', ':3.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')


    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()


    classCenter_all_s = torch.zeros(num_classes, args.bottleneck_dim).to(device)  # source accumulative class-centroids
    class_all_weight_s = torch.zeros(num_classes,1).to(device)  # source accumulative weight
    classCenter_all_t = torch.zeros(num_classes, args.bottleneck_dim).to(device)  # target accumulative class-centroids
    class_all_weight_t = torch.zeros(num_classes, 1).to(device)  # target accumulative weight

    # with torch.autograd.set_detect_anomaly(True):
    for i in range(args.iters_per_epoch):

        x_s, labels_s = next(train_source_iter)[:2]
        x_t, l_t= next(train_target_iter)[:2]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)


        # lambd = (2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1)
        lambd = (2 / (
                    1 + math.exp(-10 * (epoch * args.iters_per_epoch + i) / (args.epochs * args.iters_per_epoch))) - 1)
        classCenter_all_s, class_all_weight_s = class_center(classCenter_all_s, class_all_weight_s, f_s, y_s, device)
        classCenter_all_t, class_all_weight_t = class_center(classCenter_all_t, class_all_weight_t, f_t, y_t, device)
        transfer_loss = my_loss(classCenter_all_s, classCenter_all_t)*5 + cluster_loss(f_s,y_s, f_t,y_t,classCenter_all_t)*lambd
        # transfer_loss = my_loss(classCenter_all_s, classCenter_all_t) * 5
        # transfer_loss =  cluster_loss(f_s, y_s, f_t, y_t,classCenter_all_t) * lambd

        classCenter_all_s = classCenter_all_s.clone().detach()
        classCenter_all_t = classCenter_all_t.clone().detach()

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss + transfer_loss
        # loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        #When all examples have been trained, update source and target accumulative class-centroids, source and accumulative weight to 0.
        if i % (torch.round(instance_num/args.batch_size)) == 0:
            classCenter_all_s = torch.zeros(num_classes, args.bottleneck_dim).to(device)
            class_all_weight_s = torch.zeros(num_classes, 1).to(device)
            classCenter_all_t = torch.zeros(num_classes, args.bottleneck_dim).to(device)
            class_all_weight_t = torch.zeros(num_classes, 1).to(device)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCC for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=50, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0005, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    args = parser.parse_args()
    main(args)
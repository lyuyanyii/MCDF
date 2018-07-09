import argparse
import os
import time
import pickle
import sys
import numpy as np
import cv2
import models
import shutil
import utils
from utils import AverageMeter
import torchvision.transforms as transforms
import torchvision
import robust_loss
import datasets

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torch.autograd import Variable
import tqdm
#import matplotlib.pyplot as plt

from PIL import Image

model_names = ['Res20', 'Res110', 'Densenet', 'Densenet124', 'VGG', 'Res50']

parser = argparse.ArgumentParser( description='Robust Loss Function' )

parser.add_argument( '--arch', metavar='ARCH', choices=model_names )
parser.add_argument( '--save-folder', type=str, metavar='PATH' )
parser.add_argument( '--lr', type=float, help='initial learning rate' )
#parser.add_argument( '--lr-step', type=float, help='lr will be decayed at these steps' )
#parser.add_argument( '--lr-decay', type=float, help='lr decayed rate' )
parser.add_argument( '--data', type=str, help='the directory of data' )
parser.add_argument( '--dataset', type=str, choices=['mnist', 'cifar10', 'imgnet', 'subcifar10'] )
parser.add_argument( '--tot-iter', type=int, help='total number of iterations' )
#parser.add_argument( '--val-iter', type=int, help='do validation every val-iter steps' )
parser.add_argument( '--workers', type=int, default=4, help='number of data loading workers (default:4)' )
parser.add_argument( '-b', '--batch-size', type=int, help='mini-batch size' )
parser.add_argument( '--resume', type=str, metavar='PATH' )
parser.add_argument( '--momentum', type=float, default=0.9, help='momentum in optim' )
parser.add_argument( '--weight-decay', type=float, default=1e-4, help='weight decay' )
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument( '--evaluation',dest='evaluation',action='store_true' )
parser.add_argument( '--disable-robust', dest='disable_robust', action='store_true' )
parser.add_argument( '--alpha', type=float, default=0.1, help='weights in robust loss' )
parser.add_argument( '--moving-avg', dest='moving_avg', action='store_true' )
parser.add_argument( '--stochastic-depth', dest='sto_depth', action='store_true' )
parser.add_argument( '--depen', dest='depen', action='store_true' )
parser.add_argument( '--pretrained', dest='pretrained', action='store_true' )
parser.add_argument( '--finetune', dest='finetune', action='store_true' )
parser.add_argument( '--epoch', type=int )
parser.add_argument( '--sub-percent', dest='sub_percent', type=int )
parser.add_argument( '--seed', type=int )
parser.add_argument( '--reduce-dim', dest='reduce_dim', action='store_true' )

class Env():
    def __init__(self, args):
        self.best_acc = 0
        self.args = args

        torch.manual_seed(0)

        logger = utils.setup_logger( os.path.join( args.save_folder, 'log.log' ) )
        self.logger = logger

        for key, value in sorted( vars(args).items() ):
            logger.info( str(key) + ': ' + str(value) )

        model = getattr(models, args.arch)( sto_depth=args.sto_depth,
                                            pretrained=args.pretrained,
                                            reduce_dim=args.reduce_dim, )
        model = torch.nn.DataParallel( model ).cuda()

        """
        if args.pretrained:
            logger.info( '=> using a pre-trained model from {}'.format(args.pretrained) )
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict( checkpoint['model'] )
        else:
            logger.info( '=> initailizing the model, {}, with random weights.'.format(args.arch) )
        """
        self.model = model

        logger.info( 'Dims: {}'.format( sum([m.data.nelement() if m.requires_grad else 0
            for m in model.parameters()] ) ) )

        self.optimizer = optim.SGD( model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True )

        self.it = 0
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info( '=> loading checkpoint from {}'.format(args.resume) )
                checkpoint = torch.load( args.resume )
                self.it = checkpoint['it']
                if 'best_acc' in checkpoint.keys():
                    self.best_acc = checkpoint['best_acc']
                self.model.load_state_dict( checkpoint['model'] )
                self.optimizer.load_state_dict( checkpoint['optimizer'] )
                logger.info( '=> loaded checkpoint from {} (iter {})'.format(
                    args.resume, self.it ) )
            else:
                raise Exception("No checkpoint found. Check your resume path.")

        if args.dataset =='mnist':
            train_dataset = datasets.mnist_dataset( args.data, train=True  )
            valid_dataset = datasets.mnist_dataset( args.data, train=False )
        elif args.dataset == 'cifar10':
            ncls = 10
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
            
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=train_transform)
            valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=valid_transform)
        elif args.dataset == 'subcifar10':
            ncls = 10
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
            
            cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=train_transform)
            cifar10_valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=valid_transform)
            tot_size = 45000
            size = tot_size // 100 * args.sub_percent
            train_dataset = datasets.subdataset( cifar10_train_dataset, 0, size, args.seed )
            valid_dataset = datasets.subdataset( cifar10_valid_dataset, 45000, 50000, args.seed )
        elif args.dataset == 'imgnet':
            ncls = 1000
            args.data = '/scratch/datasets/imagenet/'
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_dataset = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            raise NotImplementedError('Dataset has not been implemented')

        ndf = model.module.ndf
        if not args.depen:
            self.criterion = robust_loss.criterion_diag( ncls, ndf, list(self.model.module.fc.parameters())[0], alp=args.alpha,
                                                moving_avg=args.moving_avg )
        else:
            self.criterion = robust_loss.criterion_mat( ncls, ndf, list(self.model.module.fc.parameters())[0], alp=args.alpha,
                                                moving_avg=args.moving_avg )
        self.criterion = torch.nn.DataParallel( self.criterion ).cuda()
        self.criterion1 = nn.CrossEntropyLoss().cuda()

        self.train_loader = data.DataLoader( train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )
        val_batch_size = args.batch_size
        self.valid_loader = data.DataLoader( valid_dataset,
            batch_size=val_batch_size, shuffle=True, num_workers=args.workers, 
            pin_memory=True, 
            )

        self.args = args
        self.save( self.best_acc )
        
        self.start_time = time.time()

        """
        args.def_iter = args.tot_iter
        if args.def_iter == 0:
            args.def_iter = args.tot_iter
        tot_epoch = (args.tot_iter - self.it) * args.batch_size // len(train_dataset) + 1
        """
        args.tot_iter = args.epoch * len(train_dataset) // args.batch_size
        if args.evaluation:
            self.valid()
        else:
            for i in range(args.epoch):
                self.train( i+1 )
                self.valid()

    def save( self, acc ):
        logger = self.logger
        is_best = acc > self.best_acc
        self.best_acc = max( self.best_acc, acc )
        logger.info( '=> saving checkpoint' )
        utils.save_checkpoint({
            'it': self.it,
            'arch': self.args.arch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best, self.args.save_folder)

    def train( self, epoch ):
        logger = self.logger
        losses = AverageMeter()

        self.model.train()
        logger.info("Classification Training Epoch {}".format(epoch) )

        if not self.args.finetune:
            if self.args.dataset == 'imgnet':
                if epoch % 30 == 0:
                    for group in self.optimizer.param_groups:
                        group['lr'] *= 0.1
            if 'cifar10' in self.args.dataset:
                if epoch == 150 or epoch == 225:
                    for group in self.optimizer.param_groups:
                        group['lr'] *= 0.1
        
        for i, batch in enumerate(self.train_loader):
            self.it += 1

            self.optimizer.zero_grad()

            inp = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

            pred, feature = self.model( inp )

            if not self.args.disable_robust and self.it > 10:
                score = self.criterion( feature, gt )
                z_k, _ = score.max(1)
                score = score - z_k[:, None, :].expand( score.size() )
                loss = ((score.exp().sum(1)+1e-6).log() + z_k).mean()
                loss1 = self.criterion1( pred, gt )
            else:
                loss = self.criterion1( pred, gt )
            losses.update( loss.data[0], inp.size(0) )
            loss.backward()
            self.optimizer.step()
            if self.it % self.args.print_freq == 0:
                log_str = 'TRAIN -> Iter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format( iter=self.it, loss=losses )
                self.logger.info( log_str )
            if self.it % 100 == 0:
                finish_time = (time.time() - self.start_time) / (self.it / self.args.tot_iter) + self.start_time
                log_str = 'Expecting finishing time {}'.format( time.asctime( time.localtime(finish_time) ) )
                self.logger.info( log_str )

    def toRGB( self, img ):
        if isinstance(img, Variable):
            img = img.type( torch.FloatTensor )
            img = img.data.numpy()
        if len(img.shape) == 2:
            img *= 255
            img = img.astype(np.uint8)
            img = cv2.applyColorMap( img, cv2.COLORMAP_JET )
        elif self.args.dataset == 'mnist':
            img = (img[0] + 0.5) * 255
        elif self.args.dataset == 'cifar10':
            img = img.transpose( 1, 2, 0 )
            mean = np.array([x/255.0 for x in [125.3, 123.0, 113.9]])
            std  = np.array([x/255.0 for x in [63.0, 62.1, 66.7]])
            img = (img * std + mean) * 255
            img = img[:, :, ::-1]
        elif self.args.dataset in ['imgnet','cub200', 'pascalvoc', 'object_discover', 'place365']:
            img = img.transpose( 1, 2, 0 )
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = img[:, :, ::-1]
            img = np.maximum( np.minimum( img, 255 ), 0)
        elif self.args.dataset == 'chestx':
            img = img.transpose( 1, 2, 0 )
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = img[:, :, ::-1]
            img = np.maximum( np.minimum( img, 255 ), 0)
        img = img.astype(np.uint8)
        return img

   
    def valid( self ):
        logger = self.logger
        self.model.eval()

        accs = AverageMeter()

        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(self.valid_loader)):
                inp = Variable( batch[0] ).cuda()
                gt  = Variable( batch[1] ).cuda()

                pred, _ = self.model( inp )
                score, pred = torch.max( pred, 1 )
                acc = (pred == gt.view(-1)).type( torch.FloatTensor ).mean()
                accs.update( acc.data[0], inp.size(0) )

        log_str = "VAL FINAL -> Accuracy: {}".format( accs.avg )
        logger.info( log_str )
        self.save( accs.avg )

if __name__ == '__main__':
    args = parser.parse_args()
    Env( args )

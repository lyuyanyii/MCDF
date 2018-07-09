import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import cv2
import numpy as np

from torch.autograd import Function, Variable

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def random_select( gt ):
    gt = gt.numpy()
    index = []
    gt2 = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] > 0:
                index.append( (i, j) )
                b = np.zeros( (20, ) )
                b[j] = 1
                gt2.append( b )
    index = np.array(index)
    gt2 = np.array(gt2)
    index = index.astype( np.int64 )
    #gt2 = Variable( torch.from_numpy(gt2) ).cuda()
    #gt2 = gt2.type( torch.cuda.FloatTensor )
    gt2 = Variable( torch.from_numpy( gt2 ), requires_grad=True ).cuda()
    gt2 = gt2.type( torch.cuda.FloatTensor )
    return index, gt2

class ToOnehot( object ):
    def __call__(self, x):
        y = torch.zeros( 20 )
        for i in range(1, 21):
            if (x == i).sum() > 0:
                y[i - 1] = 1
        y = y
        return y

def bbox_generator( img, threshold ):
    if isinstance( img, Variable ):
        img = img.type( torch.FloatTensor )
        img = img.data.numpy()
    img = (img >= threshold).astype( np.uint8 ) * 255
    im2, contours, hierarchy = cv2.findContours( img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
    size = 0
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect( contour )
        if w_ * h_ > size:
            x, y, w, h = x_, y_, w_, h_
            size = w * h
    return x, y, w, h

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def worker_init( worker_id ):
    #print(torch.initial_seed())
    torch.manual_seed(1217571572117475252)

class Quantile( Function ):
    def forward( self, x ):
        x *= 10
        x = torch.floor( x )
        x /= 10
        return x
    def backward( self, grad ):
        return grad

class Binarized( Function ):
    def forward( self, x, R ):
        #output = torch.round( x )
        output = (x > R).type( torch.cuda.FloatTensor )
        return output

    def backward( self, output_grad ):
        return output_grad, output_grad

class sharp_t( Function ):
    def forward( self, x ):
        x *= (x > 0.1).type( torch.cuda.FloatTensor )
        return x
    def backward( self, grad ):
        return grad

class ThresholdBinarized( Function ):
    def forward( self, x ):
        r = torch.rand( x.size(0), 1, 1, 1 )
        r = r.cuda().expand( x.size() )
        #x = torch.max( x, torch.ones(1).cuda() * 0.1 )
        mask_P = (x > r).type( torch.cuda.FloatTensor )
        self.save_for_backward( mask_P )
        return mask_P
    
    def backward( self, grad ):
        #mask_P, = self.saved_variables
        return grad# * mask_P.data

class Entropy( nn.Module ):
    def __init__( self ):
        super().__init__()
    def forward( self, x ):
        x = nn.Softmax()(x)
        loss = (-x * torch.log(x)).sum(1).mean(0)
        return loss

class WeightedBCELoss( nn.Module ):
    def __init__(self):
        super().__init__()
    def forward( self, input, target ):
        input = nn.Sigmoid()(input)
        """
        w0 = (target == 0).type( torch.cuda.FloatTensor )
        w1 = (target == 1).type( torch.cuda.FloatTensor )
        w0 /= w0.sum() / (w0.sum() + w1.sum()) + 1e-5
        w1 /= w1.sum() / (w0.sum() + w1.sum()) + 1e-5
        loss = -( target * torch.log(input + 1e-5) + (1 - target) * torch.log(1 - input + 1e-5) )
        loss = (loss * (w0 + w1)).mean()
        """
        input = input / (input.sum(1)[:, None].expand( input.size() ) + 1e-5)
        target = target / (target.sum(1)[:, None].expand( target.size() ) + 1e-5)
        return ((input - target)**2).sum(1).mean(0)

def cls_zero_grad( m ):
    if hasattr(m, 'cls'):
        m.zero_grad()

def weight_init( m ):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal( m.weight )
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

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

def save_checkpoint(state, is_best, save_folder, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    torch.save(state, save_folder + '/' + filename)
    if is_best:
        shutil.copyfile(save_folder + '/' + filename,
                        save_folder + '/' + 'model_best.pth.tar')


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger

class WeightsCheck():
    def __init__(self, model):
        self.params_mean = []
        dtype = torch.FloatTensor
        for param in model.parameters():
            if len(param.size()) == 4 or len(param.size()) == 5:
                self.params_mean.append(float(param.mean().type(dtype)))

    def check(self, model):
        dtype = torch.FloatTensor
        cnt = 0
        for param in model.parameters():
            if len(param.size()) == 4 or len(param.size()) == 5:
                if param.grad is None:
                    print("Warning: param with shape {} has no grad".format(param.size()))
                mean = float(param.mean().type(dtype))
                if mean == self.params_mean[cnt]:
                    print("Warning: param with shape {} has not been updated".format(param.size()))
                self.params_mean[cnt] = mean
                cnt += 1



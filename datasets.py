import torch
import torch.utils.data as data
import numpy
import random
import cv2
import sys
import torch.nn.functional as F
import gzip
import pickle
import numpy as np
import os
import torchvision.transforms as transforms

class subdataset( data.Dataset ):
    def __init__(self, dataset, a, b, seed):
        self.dataset = dataset
        self.a, self.b = a, b
        np.random.seed(seed)
        self.mapping = np.arange(0, b - a)
        if seed != -1:
            np.random.shuffle( self.mapping )

    def __len__(self):
        return self.b - self.a

    def __getitem__( self, index ):
        return self.dataset[ self.a + self.mapping[index] ]



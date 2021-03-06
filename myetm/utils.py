from __future__ import division
from __future__ import print_function

import os
import math

import torch
from vocab import Vocab
from collections import *
import numpy as np

def load_word_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path+'.txt'))
    with open(path+'.txt','r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path+'.vocab','w') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors

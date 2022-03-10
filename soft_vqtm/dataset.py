import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch
import torch.utils.data as data

from vocab import Vocab

# Dataset class for SICK dataset
class Dataset(data.Dataset):
    def __init__(self, files, args):
        super(Dataset, self).__init__()
        self.docs = files

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, lst_index):
        docs = []
        for idx in lst_index:
                doc = torch.load(self.docs[idx])[90:]
                doc = doc[doc.nonzero()]
                doc = doc.squeeze(dim=-1)
                if doc.sum() != 0:
                    docs.append(doc)
        return docs

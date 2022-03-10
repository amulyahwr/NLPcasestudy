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
        doc_tensors = []
        # doc_txts = []
        for idx in lst_index:
            doc_tensor = torch.load(self.docs[idx])

            # with open(self.docs[idx].replace('docs','mallet').replace('pt','txt'),'r') as doc_file:
            #     doc_txt = doc_file.read()

            if doc_tensor.sum() != 0:
                doc_tensors.append(doc_tensor.unsqueeze(dim=0))
                # doc_txts.append(doc_txt)


        doc_tensors = torch.cat(doc_tensors, dim=0)
        # return doc_tensors, doc_txts
        return doc_tensors

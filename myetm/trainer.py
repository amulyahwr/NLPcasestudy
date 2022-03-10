from tqdm import tqdm
import math
import torch
from torch.autograd import Variable as Var

import numpy as np
from transformers import BertTokenizer

class Trainer(object):
    def __init__(self, args, model, optimizer, vocab):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.optimizer  = optimizer
        self.epoch      = 0
        self.vocab = vocab

        self.max_len = 100


    # helper function for training
    def train(self, doc_tensors):
        self.model.train()
        self.optimizer.zero_grad()

        input_reps = doc_tensors

        #if self.args.cuda:
        #    input_reps = input_reps.cuda()


        recon_loss, kld_theta, total_tokens = self.model(input_reps, "train")

        docu_reconstr_loss = torch.sum(recon_loss)
        docu_kld_loss = kld_theta

        # print(docu_reconstr_loss)
        # print(total_tokens)
        # print(math.exp(docu_reconstr_loss/total_tokens))
        # quit()

        docu_loss = docu_reconstr_loss + docu_kld_loss

        dirichlet_loss = torch.tensor([0])
        cosine_loss = torch.tensor([0])

        docu_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return docu_reconstr_loss, docu_kld_loss, dirichlet_loss, cosine_loss, docu_loss, total_tokens

    # helper function for testing
    def test(self, doc_tensors):
        self.model.eval()

        with torch.no_grad():
            input_reps = doc_tensors

            #if self.args.cuda:
            #    input_reps = input_reps.cuda()

            recon_loss, kld_theta, total_tokens = self.model(input_reps, "test")

            docu_reconstr_loss = torch.sum(recon_loss)
            docu_kld_loss = kld_theta

            dirichlet_loss = torch.tensor([0])
            cosine_loss = torch.tensor([0])

            docu_loss = docu_reconstr_loss + docu_kld_loss

        return docu_reconstr_loss, docu_kld_loss, dirichlet_loss, cosine_loss, docu_loss, total_tokens

from tqdm import tqdm
import math
import torch
from torch.autograd import Variable as Var
import torch.nn as nn

import numpy as np

class Trainer(object):
    def __init__(self, args, model, criterion_hinge, optimizer, concepts_idxs,concepts_labels, vocab):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.optimizer  = optimizer
        self.epoch      = 0
        self.vocab = vocab

        self.criterion_hinge = criterion_hinge

        self.concepts_idxs = concepts_idxs
        self.concepts_labels = concepts_labels

        self.balance_factor = 0.001
        self.distance = nn.PairwiseDistance()

    # helper function for training
    def train(self, input_documents):
        self.model.train()
        self.optimizer.zero_grad()
        total_tokens = 0
        batch_reconstr_loss = 0.0
        batch_vq_loss = 0.0
        batch_hinge_loss = 0.0
        batch_loss = 0.0

        for docu in input_documents:

            if self.args.cuda:
                docu = docu.cuda()

            outputs, theta, vq_loss = self.model(docu, "train")

            #document length
            total_tokens = total_tokens + len(docu)

            reconstr_loss = -torch.sum(outputs)

            batch_reconstr_loss = batch_reconstr_loss + reconstr_loss.item()

            #VQ Loss
            batch_vq_loss = batch_vq_loss + vq_loss.item()

            #HingeEmbedding Loss
            concept_embeds = self.model.emb_concept(self.concepts_idxs)
            distance = self.distance(concept_embeds[:,0,:], concept_embeds[:,1,:])
            hinge_loss = self.criterion_hinge(distance, self.concepts_labels)
            hinge_loss = self.balance_factor * hinge_loss
            batch_hinge_loss = batch_hinge_loss + hinge_loss.item()


            #Total Loss
            loss = reconstr_loss + vq_loss + hinge_loss
            batch_loss = batch_loss + loss

        (batch_loss).backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_reconstr_loss, batch_vq_loss, batch_hinge_loss, batch_loss.item(), total_tokens

    # helper function for testing
    def test(self, input_documents):
        self.model.eval()

        total_tokens = 0
        batch_reconstr_loss = 0.0
        batch_vq_loss = 0.0
        batch_hinge_loss = 0.0
        batch_loss = 0.0

        with torch.no_grad():

            for docu in input_documents:

                if self.args.cuda:
                    docu = docu.cuda()

                outputs, theta, vq_loss = self.model(docu, "test")

                #document length
                total_tokens = total_tokens + len(docu)

                reconstr_loss = -torch.sum(outputs)
                batch_reconstr_loss = batch_reconstr_loss + reconstr_loss.item()

                #vq_loss = torch.tensor([0])
                batch_vq_loss = batch_vq_loss + vq_loss.item()

                #HingeEmbedding Loss
                concept_embeds = self.model.emb_concept(self.concepts_idxs)
                distance = self.distance(concept_embeds[:,0,:], concept_embeds[:,1,:])
                hinge_loss = self.criterion_hinge(distance, self.concepts_labels)
                hinge_loss = self.balance_factor * hinge_loss
                batch_hinge_loss = batch_hinge_loss + hinge_loss.item()

                #Total Loss
                loss = reconstr_loss  + vq_loss + hinge_loss
                batch_loss = batch_loss + loss

        return batch_reconstr_loss, batch_vq_loss, batch_hinge_loss, batch_loss.item(), total_tokens

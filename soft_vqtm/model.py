import torch
import torch.nn as nn
from torch.autograd import Variable as Var
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class TopicVAE(nn.Module):
    def __init__(self, args, vocab):
        super(TopicVAE, self).__init__()
        self.args = args
        self.decay = args.decay
        self.in_dim = args.in_dim
        self.hidden_size = args.hidden_size
        self.commitment_cost = args.commitment_cost
        self.vocab = vocab
        #self.max_sent_len = args.max_sent_len
        self.numbr_concepts = args.numbr_concepts

        #Embeddings
        #self.embedding_dropout = nn.Dropout(p=0.2)
        self.emb = nn.Embedding(len(self.vocab), self.in_dim)
        self.emb.weight.requires_grad = False

        self.emb_concept = nn.Embedding(self.numbr_concepts, self.in_dim)
        self.emb_concept.weight.data.uniform_(-1/self.numbr_concepts, 1/self.numbr_concepts)

        self.quant2vocab = nn.Linear(self.in_dim, len(self.vocab))

    def vqvae(self, embedded_input, mode):
        # Calculate dot product
        dot_prod = torch.matmul(embedded_input, self.emb_concept.weight.t())

        #Topic Proportion
        thetas = torch.softmax(dot_prod, dim=1)

        theta = torch.sum(thetas, dim=0, keepdim=True)
        theta = theta/torch.sum(theta)
        #print(theta)
        #quit()

        #Quantization
        quantized_words = torch.matmul(thetas, self.emb_concept.weight)

        # Loss
        e_latent_loss = torch.mean((quantized_words.detach() - embedded_input)**2)
        q_latent_loss = torch.mean((quantized_words - embedded_input.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized_words = embedded_input + (quantized_words - embedded_input).detach()

        quantized_docu = torch.mean(quantized_words, dim=0, keepdim=True)

        return thetas, theta, quantized_words, quantized_docu, loss

    def forward(self, input_document, mode):

        #embeddig of document
        embedded_document = self.emb(input_document)

        #VQ-VAE
        thetas, theta, quantized_words, quantized_docu, vq_loss = self.vqvae(embedded_document, mode)

        #quant2vocab
        res = self.quant2vocab(quantized_docu)
        res = torch.softmax(res, dim=1)
        outputs = torch.log(res+1e-6)

        #bin-count
        bin_count = torch.bincount(input_document, minlength=len(self.vocab)).float()

        outputs = outputs * bin_count

        return outputs, theta, vq_loss

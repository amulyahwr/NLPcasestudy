import torch
import torch.nn as nn
from torch.autograd import Variable as Var
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class ETM(nn.Module):
    def __init__(self, args, vocab):
        super(ETM, self).__init__()
        ## define hyperparameters
        self.args = args
        self.hidden_size = args.hidden_size
        self.vocab = vocab
        self.bow_norm = args.bow_norm
        self.numbr_concepts = args.numbr_concepts
        self.vocab_size = len(self.vocab)

        self.rho_size = args.in_dim
        self.enc_drop = args.enc_drop
        self.emsize = args.in_dim
        self.t_drop = nn.Dropout(self.enc_drop)

        self.theta_act = self.get_activation('relu')

        ## define the word embedding matrix \rho
        self.rho = nn.Embedding(self.vocab_size, self.rho_size)
        self.rho.weight.requires_grad = False

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(self.rho_size, self.numbr_concepts, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(self.vocab_size, self.hidden_size),
                self.theta_act,
                nn.Linear(self.hidden_size, self.hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(self.hidden_size, self.numbr_concepts, bias=True)
        self.logsigma_q_theta = nn.Linear(self.hidden_size, self.numbr_concepts, bias=True)

    def convert2bow(self, input_reps, mode):

        bincount_documents = []
        bincount_documents_norm = []
        total_tokens = 0
        for input_rep in input_reps:
            input_rep = torch.bincount(input_rep, minlength = len(self.vocab))
            input_rep[0] = 0
            total_tokens = total_tokens + torch.sum(input_rep)
            input_rep = torch.unsqueeze(input_rep.float(), dim=0)
            if self.bow_norm:
                 input_rep_norm = input_rep/torch.sum(input_rep)
            bincount_documents.append(input_rep)
            bincount_documents_norm.append(input_rep_norm)

        bincount_documents = torch.cat(bincount_documents, dim=0)
        bincount_documents_norm = torch.cat(bincount_documents_norm, dim=0)
        if self.args.cuda:
            bincount_documents = bincount_documents.cuda()
            bincount_documents_norm = bincount_documents_norm.cuda()
        return bincount_documents,bincount_documents_norm, total_tokens


    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar, mode):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if mode=='train':
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        # print(bows.sum())
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        # print(mu_theta.sum())
        # print(logsigma_theta.sum())

        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows, mode):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta, mode)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res+1e-6)
        return preds

    def forward(self, input_reps, mode):
        bincount_documents, bincount_documents_norm, total_tokens = self.convert2bow(input_reps, mode)
        # print(bincount_documents)
        # print(total_tokens)
        # quit()
        theta, kld_theta = self.get_theta(bincount_documents_norm, mode)

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        # print(preds.sum())
        # print(bincount_documents.sum())
        recon_loss = -(preds * bincount_documents).sum(1)

        # print(preds.shape)
        # print(bincount_documents.shape)
        # print(recon_loss)
        # print(kld_theta)
        return recon_loss, kld_theta, total_tokens

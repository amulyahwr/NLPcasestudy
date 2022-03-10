from copy import deepcopy
import torch
import torch.nn as nn

import torch.nn.functional as F

from models.reg_lstm.weight_drop import WeightDrop
from models.reg_lstm.embed_regularize import embedded_dropout
import pickle
import numpy as np

class RegLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        target_class = config.target_class
        self.is_bidirectional = config.bidirectional
        self.has_bottleneck_layer = config.bottleneck_layer
        self.mode = config.mode
        self.tar = config.tar
        self.ar = config.ar
        self.beta_ema = config.beta_ema  # Temporal averaging
        self.wdrop = config.wdrop  # Weight dropping
        self.embed_droprate = config.embed_droprate  # Embedding dropout
        with open("/research4/projects/topic_modeling_autoencoding/fidgit/hedwig/models/reg_lstm/model_0_concept_embed.pickle", 'rb') as pickle_file:
            self.concept_embeddings = pickle.load(pickle_file).cuda()
            self.concept_embeddings.requires_grad = False
        
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(config.words_num, config.words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported Mode")
            exit()

        self.lstm = nn.LSTM(config.words_dim, config.hidden_dim, dropout=config.dropout, num_layers=config.num_layers,
                            bidirectional=self.is_bidirectional, batch_first=True)

        if self.wdrop:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)
        self.dropout = nn.Dropout(config.dropout)

        if self.has_bottleneck_layer:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config.hidden_dim, config.hidden_dim)  # Hidden Bottleneck Layer
                self.fc2 = nn.Linear(config.hidden_dim, target_class)
            else:
                self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim//2)   # Hidden Bottleneck Layer
                self.fc2 = nn.Linear(config.hidden_dim//2, target_class)
        else:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config.hidden_dim, target_class)
            else:
                self.fc1 = nn.Linear(config.hidden_dim, target_class)
        
        if self.beta_ema>0:
            self.avg_param = deepcopy(list((name, p) for name, p in self.named_parameters()))
            if torch.cuda.is_available():
                self.avg_param = [(a[0], a[1].cuda()) for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x, lengths=None):
        if self.mode == 'rand':
            x = embedded_dropout(self.embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.embed(x)
        elif self.mode == 'static':
            x = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
        elif self.mode == 'non-static':
            x = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
        else:
            print("Unsupported Mode")
            exit()
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_outs, _ = self.lstm(x)
        rnn_outs_temp = rnn_outs

        if lengths is not None:
            rnn_outs,_ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)
            rnn_outs_temp, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs_temp, batch_first=True)

        #vqvae
        embedded_concepts = self.concept_embeddings.unsqueeze(dim=0).repeat(rnn_outs_temp.shape[0], 1, 1)
        dot_prod = torch.einsum('abc, adc -> abd', rnn_outs_temp, embedded_concepts)

        #Topic Proportion
        thetas = torch.softmax(dot_prod, dim=-1)

        theta = torch.sum(thetas, dim=1)
        theta = theta/torch.sum(theta, dim=-1, keepdim=True)

        #Quantization
        quantized_words = torch.einsum('abc, acd -> abd', thetas, embedded_concepts)
        # rnn_outs_temp = torch.cat((rnn_outs_temp, quantized_words), dim=-1)
        rnn_outs_temp = (rnn_outs_temp + quantized_words)/2
        x = F.relu(torch.transpose(rnn_outs_temp, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        if self.has_bottleneck_layer:
            x = F.relu(self.fc1(x))
            # x = self.dropout(x)
            if self.tar or self.ar:
                return self.fc2(x), rnn_outs.permute(1,0,2)
            return self.fc2(x)
        else:
            if self.tar or self.ar:
                return self.fc1(x), rnn_outs.permute(1,0,2)
            return self.fc1(x)

    def update_ema(self):
        self.steps_ema += 1
        params = {}
        new_avg_p = []
        for name, param in self.named_parameters():
            params[name] = param.data
        for name, avg_p in self.avg_param:
            avg_p = avg_p * self.beta_ema + (1-self.beta_ema) * params[name].data
            new_avg_p.append((name, avg_p))
        self.avg_param = new_avg_p
    
    def load_ema_params(self):
        avg_params = {}
        for name, param in self.avg_param:
            avg_params[name] = param.data

        for name, p in self.named_parameters():
            if name in avg_params.keys():
                p.data.copy_(avg_params[name]/(1-self.beta_ema**self.steps_ema))
        # for p, avg_p in zip(self.parameters(), self.avg_param):
        #     p.data.copy_(avg_p/(1-self.beta_ema**self.steps_ema))
            
    def load_params(self, params):
        for p,avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)
            
    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

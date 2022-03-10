from __future__ import division
from __future__ import print_function

import logging
import os
import torch
import random
import operator

import torch.optim as optim
from tqdm import tqdm
# UTILITY FUNCTIONS
from utils import load_word_vectors
# CONFIG PARSER
from config import parse_args

# DATASET CLASS FOR DATASET
from dataset import Dataset
# NEURAL NETWORK MODULES/LAYERS
from model import *
# TRAIN HELPER FUNCTION
from trainer import Trainer
import numpy as np
import pickle as pk
import pandas as pd
import math
import glob
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname) + '.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()

    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        print("Using CUDA")
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    vocab = os.path.join(args.data, 'vocab.pkl')
    vocab = pk.load(open(vocab,'rb'))

    vocab_size=len(vocab)
    print("Vocab Size: %d"%(vocab_size))

    train_files = glob.glob(os.path.join(args.data, 'train_docs/*.pt'))
    train_dataset = Dataset(train_files, args)
    logger.debug('==> Size of train data   : %d ' % len(train_files))

    dev_files = glob.glob(os.path.join(args.data, 'dev_docs/*.pt'))
    dev_dataset = Dataset(dev_files, args)
    logger.debug('==> Size of dev data   : %d ' % len(dev_files))

    test_files = glob.glob(os.path.join(args.data, 'test_docs/*.pt'))
    test_dataset = Dataset(test_files, args)
    logger.debug('==> Size of test data   : %d ' % len(test_files))

    #Concept embeddings
    concepts_idxs_file = os.path.join(args.data, 'concepts_idxs_%d.pth'%(args.numbr_concepts))
    concepts_labels_file = os.path.join(args.data, 'concepts_labels_%d.pth'%(args.numbr_concepts))
    if os.path.isfile(concepts_idxs_file):
        concepts_idxs = torch.load(concepts_idxs_file)
        concepts_labels = torch.load(concepts_labels_file)
    else:
        concepts_idxs = []
        concepts_labels = []
        for i in range(args.numbr_concepts):
            for j in range(args.numbr_concepts):
                if i == j:
                    concepts_idxs.append(torch.unsqueeze(torch.LongTensor(np.array([i, j])),dim=0))
                    concepts_labels.append(1)
                else:
                    concepts_idxs.append(torch.unsqueeze(torch.LongTensor(np.array([i, j])),dim=0))
                    concepts_labels.append(-1)

        concepts_idxs = torch.cat(concepts_idxs, dim=0)
        concepts_labels = torch.FloatTensor(np.array(concepts_labels))

        torch.save(concepts_idxs, concepts_idxs_file)
        torch.save(concepts_labels, concepts_labels_file)

    if args.cuda:
        concepts_idxs = concepts_idxs.cuda()
        concepts_labels = concepts_labels.cuda()
    #initialize model, optimizer
    model = TopicVAE(args, vocab)
    criterion_cosine = nn.HingeEmbeddingLoss(reduction='mean',margin=1)

    if args.cuda:
        model.cuda(), criterion_cosine.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, amsgrad=True)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=args.lr)

    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.6B.300d'))

        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(len(vocab), args.in_dim).normal_(-0.05, 0.05)

        emb[0].zero_()
        emb[1].zero_()
        emb[2].zero_()
        emb[3].zero_()

        for word in vocab.keys():
            if glove_vocab.getIndex(word):
                emb[vocab[word]] = glove_emb[glove_vocab.getIndex(word)]

        torch.save(emb, emb_file)
        
    if args.cuda:
        emb = emb.cuda()
    model.emb.weight.data.copy_(emb)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion_cosine, optimizer, concepts_idxs, concepts_labels, vocab)

    best_ppl = float('inf')
    best_loss = float('inf')

    early_stop_count = 0
    columns = ['ExpName','ExpNo', 'Epoch', 'ReconstLoss','TotalVQLoss', 'TotalConceptLoss','TotalLoss', 'Perplexity']
    results = []

    train_idx = list(np.arange(len(train_dataset)))
    dev_idx = list(np.arange(len(dev_dataset)))
    test_idx = list(np.arange(len(test_dataset)))

    for epoch in range(args.epochs):
        train_total_reconstr_loss = 0.0
        train_total_vq_loss = 0.0
        train_total_concept_space_loss = 0.0
        train_total_loss = 0.0
        train_total_tokens = 0.0

        dev_total_reconstr_loss = 0.0
        dev_total_vq_loss = 0.0
        dev_total_concept_space_loss = 0.0
        dev_total_loss = 0.0
        dev_total_tokens = 0.0

        test_total_reconstr_loss = 0.0
        test_total_vq_loss = 0.0
        test_total_concept_space_loss = 0.0
        test_total_loss = 0.0
        test_total_tokens = 0.0

        random.shuffle(train_idx)
        random.shuffle(dev_idx)
        random.shuffle(test_idx)

        batch_train_data = [train_idx[i:i + args.batchsize] for i in range(0, len(train_idx), args.batchsize)]
        batch_dev_data = [dev_idx[i:i + args.batchsize] for i in range(0, len(dev_idx), args.batchsize)]
        batch_test_data = [test_idx[i:i + args.batchsize] for i in range(0, len(test_idx), args.batchsize)]

        for batch in tqdm(batch_train_data, desc='Training batches..'):
            input_documents = train_dataset[batch]

            train_batch_reconstr_loss, \
            train_batch_vq_loss, \
            train_batch_concept_space_loss, \
            train_batch_loss, \
            train_batch_tokens = trainer.train(input_documents)

            train_total_reconstr_loss = train_total_reconstr_loss + train_batch_reconstr_loss
            train_total_vq_loss = train_total_vq_loss + train_batch_vq_loss
            train_total_concept_space_loss = train_total_concept_space_loss + train_batch_concept_space_loss
            train_total_loss = train_total_loss + train_batch_loss
            train_total_tokens = train_total_tokens + train_batch_tokens

        train_total_perplexity = math.exp(train_total_reconstr_loss/train_total_tokens)

        for batch in tqdm(batch_dev_data, desc='Dev batches..'):
            input_documents = dev_dataset[batch]

            dev_batch_reconstr_loss, \
            dev_batch_vq_loss, \
            dev_batch_concept_space_loss, \
            dev_batch_loss, \
            dev_batch_tokens = trainer.test(input_documents)

            dev_total_reconstr_loss = dev_total_reconstr_loss + dev_batch_reconstr_loss
            dev_total_vq_loss = dev_total_vq_loss + dev_batch_vq_loss
            dev_total_concept_space_loss = dev_total_concept_space_loss + dev_batch_concept_space_loss
            dev_total_loss = dev_total_loss + dev_batch_loss
            dev_total_tokens = dev_total_tokens + dev_batch_tokens

        dev_total_perplexity = math.exp(dev_total_reconstr_loss/ dev_total_tokens)

        for batch in tqdm(batch_test_data, desc='Testing batches..'):
            input_documents = test_dataset[batch]

            test_batch_reconstr_loss, \
            test_batch_vq_loss, \
            test_batch_concept_space_loss, \
            test_batch_loss, \
            test_batch_tokens = trainer.test(input_documents)

            test_total_reconstr_loss = test_total_reconstr_loss + test_batch_reconstr_loss
            test_total_vq_loss = test_total_vq_loss + test_batch_vq_loss
            test_total_concept_space_loss = test_total_concept_space_loss + test_batch_concept_space_loss
            test_total_loss = test_total_loss + test_batch_loss
            test_total_tokens = test_total_tokens + test_batch_tokens

        test_total_perplexity = math.exp(test_total_reconstr_loss/test_total_tokens)

        logger.info('==> Training Epoch {}, \
                        \nReconstruction Loss: {}, \
                        \nVQ Loss: {}, \
                        \nConcept Loss: {}, \
                        \nTotal Loss: {}, \
                        \nPerplexity: {}'.format(epoch + 1, \
                                            train_total_reconstr_loss/(len(batch_train_data) * args.batchsize), \
                                            train_total_vq_loss/(len(batch_train_data) * args.batchsize), \
                                            train_total_concept_space_loss/(len(batch_train_data) * args.batchsize), \
                                            train_total_loss/(len(batch_train_data) * args.batchsize), \
                                            train_total_perplexity))
        logger.info('==> Dev Epoch {}, \
                        \nReconstruction Loss: {}, \
                        \nVQ Loss: {}, \
                        \nConcept Loss: {}, \
                        \nTotal Loss: {}, \
                        \nPerplexity: {}'.format(epoch + 1, \
                                            dev_total_reconstr_loss/(len(batch_dev_data) * args.batchsize), \
                                            dev_total_vq_loss/(len(batch_dev_data) * args.batchsize), \
                                            dev_total_concept_space_loss/(len(batch_dev_data) * args.batchsize), \
                                            dev_total_loss/(len(batch_dev_data) * args.batchsize), \
                                            dev_total_perplexity))

        logger.info('==> Testing Epoch {}, \
                        \nReconstruction Loss: {}, \
                        \nVQ Loss: {}, \
                        \nConcept Loss: {}, \
                        \nTotal Loss: {}, \
                        \nPerplexity: {}'.format(epoch + 1, \
                                            test_total_reconstr_loss/(len(batch_test_data) * args.batchsize), \
                                            test_total_vq_loss/(len(batch_test_data) * args.batchsize), \
                                            test_total_concept_space_loss/(len(batch_test_data) * args.batchsize), \
                                            test_total_loss/(len(batch_test_data) * args.batchsize), \
                                            test_total_perplexity))
        #quit()
        results.append((args.expname, \
                        args.expno, \
                        epoch+1, \
                        test_total_reconstr_loss/(len(batch_test_data) * args.batchsize), \
                        test_total_vq_loss/(len(batch_test_data) * args.batchsize), \
                        test_total_concept_space_loss/(len(batch_test_data) * args.batchsize), \
                        test_total_loss/(len(batch_test_data) * args.batchsize), \
                        test_total_perplexity))

        if (best_loss > dev_total_loss):
            best_loss = dev_total_loss

            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'args': args, 'epoch': epoch}
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % (os.path.join(args.save, args.expname+str(args.expno))))

            with open("./results/%s%d_concept_embed.pickle"%(args.expname, args.expno), "wb") as ce:
                pk.dump(model.emb_concept.weight.data, ce)


            csv_data = np.array(results)
            df = pd.DataFrame(csv_data, columns = columns)
            df.to_csv(("./results/%s%d_results.csv"%(args.expname, args.expno)), index = False)

        else:
            early_stop_count = early_stop_count + 1

            if early_stop_count == 20:
                quit()


if __name__ == "__main__":
    main()

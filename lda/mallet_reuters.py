import os
import numpy as np
import pickle as pk
from tqdm import tqdm
dataset = 'reuters_output'

train_dir = '../../fidgit/data/train_mallet/'
dev_dir = '../../fidgit/data/dev_mallet/'
test_dir = '../../fidgit/data/test_mallet/'

num_topics = 90

#create mallet format for training data
os.system('/research2/tools/mallet/bin/mallet import-dir \
           --input %s \
           --output ./%s/train_%d.mallet \
           --keep-sequence TRUE \
           --remove-stopwords TRUE'%(train_dir, dataset, num_topics))

#run mallet for training data
os.system('/research2/tools/mallet/bin/mallet train-topics \
           --input ./%s/train_%d.mallet \
           --evaluator-filename ./%s/evaluate_%d.mallet \
           --inferencer-filename ./%s/inference_%d.mallet \
           --show-topics-interval 10000 \
           --num-topics %d \
           --num-top-words 10 \
           --num-iterations 1000 \
           --diagnostics-file ./%s/train_diag_%d.txt \
           --topic-word-weights-file ./%s/train_tww_%d.txt \
           --word-topic-counts-file ./%s/train_wtc_%d.txt \
           --output-doc-topics ./%s/train_doc_topics_%d.txt \
           --output-topic-keys ./%s/train_topics_keys_%d.txt \
           --random-seed 1'%(dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics))

#create mallet format for test data
os.system('/research2/tools/mallet/bin/mallet import-dir \
           --input %s \
           --output ./%s/test_%d.mallet \
           --keep-sequence TRUE \
           --remove-stopwords TRUE \
           --use-pipe-from ./%s/train_%d.mallet'%(test_dir, dataset,num_topics, dataset, num_topics))

#run mallet for test data
os.system('/research2/tools/mallet/bin/mallet evaluate-topics \
           --input ./%s/test_%d.mallet \
           --evaluator ./%s/evaluate_%d.mallet \
           --output-doc-probs ./%s/test_doc_probs_%d.txt \
           --output-prob ./%s/test_probs_%d.txt \
           --num-iterations 1000 \
           --random-seed 1'%(dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics,
                               dataset,
                               num_topics))

#get doc-topics for test dataset
os.system('/research2/tools/mallet/bin/mallet infer-topics \
            --input ./%s/test_%d.mallet \
            --inferencer ./%s/inference_%d.mallet \
            --output-doc-topics ./%s/test_doc_topics_%d.txt \
            --random-seed 1'%(dataset,
                                num_topics,
                                dataset,
                                num_topics,
                                dataset,
                                num_topics))

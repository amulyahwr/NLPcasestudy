import pickle
import torch
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

expno = 0

vocab = '../data/vocab.pkl'
vocab = pickle.load(open(vocab,'rb'))
vocab_size=len(vocab)

concepts_emb = pickle.load(open('./results/model_%d_concept_embed.pickle'%(expno),'rb')).cpu()
emb_file = os.path.join('../data/', 'embed.pth')
word_emb = torch.load(emb_file)

vocab_dist = torch.softmax(torch.matmul(concepts_emb, word_emb.t()), dim=1)
topk_values, topk_indices = torch.topk(vocab_dist,10, dim=1)

def topic_coherence(expno, vocab, vocab_dist, topk_values, topk_indices):
    print("Calculating topic coherence...")
    if not os.path.isfile("model_%d_top_words.txt"%(expno)):

        str = ''
        for idx in tqdm(topk_indices):
            words = []
            for word_idx in idx:
                words.append(list(vocab.keys())[list(vocab.values()).index(word_idx)])

            str = str + ' '.join(words) + '\n'

        text_file = open("model_%d_top_words.txt"%(expno), "w")
        text_file.write(str)
        text_file.close()

    cmd = "java -jar /research2/tools/palmetto/palmetto-0.1.0-jar-with-dependencies.jar /research2/tools/palmetto/wikipedia_bd C_V model_%d_top_words.txt"%(expno)
    os.system(cmd)
    print("*"*50)

def cosine_similarity(concepts_emb):
    print("Calculating cosine similairty between topic embeddings...")
    d = torch.matmul(concepts_emb, concepts_emb.t())
    norm = (concepts_emb * concepts_emb).sum(1, keepdims=True) ** .5


    cs = d / norm / norm.t()
    total_cs = torch.sum(torch.triu(cs, diagonal=1))
    print("Total Cosine Similairty: %f"%(total_cs))
    print("*"*50)
    return total_cs

def distance_similarity(concepts_emb):
    print("Calculating distance similairty between topic embeddings...")
    d = euclidean_distances(concepts_emb, concepts_emb, squared=True)
    d = torch.from_numpy(d)
    d = d / torch.max(d)

    total_cs = torch.sum(torch.triu(d, diagonal=1))
    print("Total Distance Similairty: %f"%(total_cs/((d.shape[0]*d.shape[0]/2)-d.shape[0])))
    print("*"*50)
    return total_cs

def topic_diversity(vocab_dist, topk_values, topk_indices):
    print("Calculating topic diveristy...")
    num_topics = vocab_dist.shape[0]
    n_unique = len(torch.unique(topk_indices))
    TD = n_unique / (topk_values.shape[1] * num_topics)
    print('Topic diveristy is: %f'%(TD))
    print("*"*50)

#Evaulate
topic_coherence(expno, vocab,  vocab_dist, topk_values, topk_indices)
# cosine_similarity(concepts_emb)
topic_diversity(vocab_dist, topk_values, topk_indices)
#distance_similarity(concepts_emb)
